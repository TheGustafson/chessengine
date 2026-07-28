# 06 — Iterative deepening, quiescence, and aspiration

Alpha-beta from file 05 is a correct search skeleton, but three things stand
between it and a strong engine. First, an engine does not get to pick a depth and
compute forever — it has a clock, and it must always have a move ready. Second, a
fixed-depth search gets *fooled* at its own leaves: it stops in the middle of a
capture sequence and mis-evaluates a position that is about to change violently.
Third, the root search wastes effort exploring a window far wider than it needs.
The three fixes — **iterative deepening**, **quiescence search**, and **aspiration
windows** — are the subject of this file. Each is a small idea with an outsized
effect, and together they are what "iterative-deepening alpha-beta" means in
practice.

## Iterative deepening: search depth 1, then 2, then 3, …

The obvious way to search to depth `N` is to call the search with depth `N`. The
counterintuitive truth is that it is *better* to search to depth 1, then throw
that away and search to depth 2, then throw *that* away and search to depth 3, and
so on up to `N` — **iterative deepening**. You redo all the shallow work every
time. It sounds wasteful. It is faster anyway, and it gives you two things you
cannot live without.

**Why it is not wasteful.** Because the tree grows like `√b` per ply (file 05,
with good ordering), the cost of a search is dominated by its *last* ply. A depth-`N`
search costs roughly `c · b^(N/2)`; all the shallower searches 1…N−1 together cost
roughly `c · b^(N/2) / (√b − 1)`, a fraction of the final search. For `b ≈ 35`,
`√b ≈ 5.9`, so the entire overhead of all previous iterations is about
`1/(5.9−1) ≈ 20%`. You pay a ~20% tax on the shallow re-searches — and get back far
more than 20% from the two benefits below.

**Benefit 1: move-ordering feedback.** This is the big one, and it ties directly to
file 05's central fact that ordering is everything. When you finish the depth-`k`
search, you know the best move (and the best line, the *principal variation*) at
that depth. That move is an excellent guess for the best move at depth `k+1`, so you
search it *first* at the next iteration. The transposition table (file 07) also
carries per-position best moves down from the shallow search to the deep one. The
result: each iteration hands the next one near-perfect move ordering, which is
exactly the condition that makes alpha-beta and PVS achieve their `b^(d/2)` best
case. A cold depth-`N` search has none of this ordering information and would
prune far worse — so much worse that it typically searches *more* nodes than the
full iterative sequence 1…N. Iterative deepening is not a compromise you make for
the clock; it is genuinely the faster way to reach depth `N`.

**Benefit 2: anytime behavior.** An engine with a clock must always have a legal
move ready to play the instant time runs out. Iterative deepening gives you exactly
that: after each completed iteration you have a fully-searched best move at that
depth. When the clock expires mid-iteration, you play the best move from the *last
completed* iteration — a real, fully-validated answer, just one ply shallower than
where you were cut off. There is no "half a search" to salvage. This is what
`Worker::iterative_deepening` in `src/search/mod.rs` drives: a loop that deepens
one ply at a time, checking the clock between iterations, until time or a depth
limit stops it. File 13 covers how the time decision is actually made; here the
point is that iterative deepening is what *makes* graceful time management
possible.

## The horizon effect and why fixed depth lies

Now the second problem. Suppose the search stops at depth 6, and at one of its
depth-6 leaves the position is: White has just captured a pawn with a queen, and
Black is about to recapture the queen with a pawn. A fixed-depth search calls
`evaluate` on that leaf and sees White up a pawn — because the recapture is one ply
beyond the horizon. White "wins a pawn" in the search's eyes, when in reality White
is about to lose a queen. The search has been fooled by stopping in the middle of
a tactical exchange.

This is the **horizon effect**: a static evaluation is only trustworthy on a
*quiet* position, one with no pending captures or checks that would swing the
material balance. Call `evaluate` on a volatile position and you get nonsense.
Worse, a naive engine will actively *use* the horizon to hide bad news: facing an
unavoidable loss of material at depth `d`, it will throw in pointless checks or
sacrifices to push the loss past depth `d`, "solving" the problem by not looking at
it. The horizon effect is not a minor inaccuracy; it makes fixed-depth search
tactically blind.

To make it concrete, picture a depth-2 search of a single exchange. White plays
`Nxp` — knight takes pawn — and the search, at its depth-2 horizon, evaluates the
resulting position: White is up one pawn (roughly +100). The search records "this
line wins a pawn" and may well choose it. But the very next ply, just past the
horizon, is `…pxN` — a pawn recaptures the knight. The true outcome of `Nxp` was
not "win a pawn" but "lose a knight for a pawn," about −200. The search's number
had the *wrong sign of a big margin*, purely because it stopped counting one ply
too early, in the middle of a trade. No amount of extra depth fixes this in
general — every fixed horizon has *some* position mid-exchange sitting on it.

## Quiescence search

The fix is to never evaluate a volatile position. When the main search reaches its
depth limit, instead of calling `evaluate` directly, it calls a special
**quiescence search** ("qsearch") that keeps going — but only through the
"noisy" moves — until the position is *quiet*, and evaluates there. Quiescence is a
second, tiny alpha-beta search with a restricted move set. ARCHON's lives in the
qsearch path of `src/search/mod.rs`.

The moves qsearch considers are, principally, **captures** (and some checks and
queen promotions) — the moves that change material. It ignores quiet moves, because
quiet moves rarely swing the evaluation the way an unresolved capture does, and
including them would make qsearch explode into a full search. By resolving all the
pending captures, qsearch reaches positions where the static evaluation is actually
meaningful.

Two mechanisms make qsearch terminate and stay cheap:

**Stand-pat.** In the main search, the side to move must make a move. In quiescence,
we assume the side to move is *not obligated* to capture — they could just "stand
pat" and accept the current static evaluation, because it is (usually) not forced to
worsen its position by capturing. So qsearch first evaluates the position statically
(the "stand-pat" score) and treats it as a lower bound on what the side to move can
achieve: if the stand-pat score already exceeds beta, we fail high and return
immediately without searching any captures. Only if the static score is *not* good
enough does qsearch try captures to improve it. This gives qsearch a natural floor
and makes it terminate: each capture removes material from the board, so the
capture chains are short.

**Delta-style pruning.** Even among captures, qsearch prunes the hopeless ones.
If the side to move is so far behind that even winning the piece being captured
(plus a margin) would not bring the score near alpha, the capture is not worth
searching — this is "futility" pruning inside qsearch, gated on a
`futility_base = static_eval + margin`. ARCHON also uses static exchange evaluation
(SEE, from file 04 — does this capture sequence actually win material?) to skip
captures that lose material outright. These pruning rules keep qsearch from
searching obviously-pointless captures, which matters because qsearch runs at
*every* leaf of the main search and its cost is a large fraction of total node
count.

One subtlety worth noting: when the side to move is *in check*, standing pat is not
allowed (you cannot "pass" out of check), so qsearch in check generates all
evasions and prunes nothing until at least one legal escape has produced a real
value. ARCHON's qsearch handles that in-check case specially, and even runs a
narrow stalemate check to catch the rare "won material straight into a stalemate"
trap. The details are engineering; the concept is: **quiescence search extends the
main search through forcing moves so that evaluation is only ever called on quiet
positions.** Without it, an engine is tactically hopeless; with it, the leaf values
that alpha-beta propagates are trustworthy.

## Aspiration windows

The third refinement targets the *root* window. File 05's search starts at the root
with the maximal window `[−∞, +∞]`. But after several iterations of deepening, you
have a very good guess for what the score will be: it will be close to last
iteration's score, because one more ply rarely changes the evaluation dramatically.
Searching with the full `[−∞, +∞]` window ignores that knowledge and does more work
than necessary — a wide window means fewer cutoffs.

An **aspiration window** narrows the root search to a guess: search with
`[guess − δ, guess + δ]` for a small delta `δ`, betting the true score lands inside.
A narrow window prunes far more aggressively (more nodes exceed the tight bounds and
get cut), so when the bet pays off, the iteration is much cheaper. ARCHON centers the
window on a running average of recent scores and sizes delta partly from the score's
variance (see the aspiration description in the `src/search/mod.rs` header; roughly
`5 + |mean_squared_score| / 10588` internal units).

The catch: sometimes the true score lands *outside* the narrow window, and the search
"fails" — it returns a bound instead of an exact value:

- **Fail-high at the root** (score ≥ the window's beta): the position is better than
  we guessed. We do not have the exact value, only that it is at least beta. So we
  widen the window upward — push beta out — and re-search.
- **Fail-low at the root** (score ≤ alpha): worse than we guessed. Widen downward and
  re-search.

Each re-search widens delta (ARCHON grows it by about ×1.34 each time), so a badly
mis-estimated score converges in a few tries, and a wildly wrong one eventually
re-opens to the full window. The economics: when the guess is good — which is *most*
of the time, because scores are stable between iterations — you pay for one cheap
narrow search; when it is off, you pay for the narrow search *plus* a re-search, a
net loss for that iteration. Aspiration windows win because the good case is common
and cheap and the bad case is rare. ARCHON layers a couple of refinements on top —
after a fail-high it runs each consecutive re-search one ply shallower (the score is
established; spend less confirming it), and it clamps the reported score to the
window on a bound so the info shown to the user stays honest.

## Search instability

Aspiration windows, PVS re-searches, and the transposition table together introduce
a phenomenon worth naming because it surprises people reading engine output:
**search instability**. Because the same position can be searched with different
windows at different times (a narrow scout, then a wide re-search; a shallow TT
entry, then a deeper visit), the score for a move can *change* — even decrease —
between and within iterations, and the "best move" can flip back and forth as the
search deepens. A depth-10 search might prefer move A, depth 11 prefer move B, and
depth 12 return to A.

This is not a bug. It is the search discovering that a line it liked at shallow
depth has a deeper refutation, or vice versa. The value returned by a search under a
narrow window is only a bound, not an exact value (file 05's fail-high/fail-low), so
comparing a bound from one search against an exact value from another is comparing
different kinds of claim. ARCHON's search takes care to handle the fallout — for
instance, it protects a *proven* mate from being overwritten by a weaker
re-scored value, and it repairs the rare case where an aborted iteration returns an
untrustworthy losing score (the `root_hygiene` logic in `src/search/mod.rs`).
Understanding that scores are windowed, bounded claims — not stable ground truth —
is the mental model that makes engine output (and engine code) stop being confusing.

## How the three fit together

Here is one "think about this position" cycle, top to bottom:

```
for depth in 1, 2, 3, ... until time runs out:        # iterative deepening
    guess = running average of recent root scores
    delta = small, based on score variance
    loop:                                             # aspiration
        window = [guess - delta, guess + delta]
        score  = alpha_beta(root, window, depth)      # PVS inside (file 05)
            # ... which, at every leaf, calls ...
            qsearch(...)                              # quiescence: resolve captures
        if score inside window: break
        widen delta in the direction of the fail; re-search
    record best move and PV; feed them to the next iteration's ordering
play the best move from the last completed iteration                # anytime
```

Iterative deepening is the outer loop that provides ordering feedback and a
ready move. Aspiration windows narrow each iteration's root search. PVS (file 05)
does the pruning inside. Quiescence guards every leaf so the propagated values are
trustworthy. Remove any one and the engine gets dramatically weaker: without
iterative deepening you lose ordering and time control; without quiescence you are
tactically blind; without aspiration you leave nodes on the table. The
transposition table (file 07) threads through all of them, carrying results and
best-moves between iterations and between transposing paths.

## What you now know

You know iterative deepening: search depths 1, 2, 3, … in turn, which is faster
than jumping to depth `N` because the shallow iterations cost a ~20% tax while
handing the deep search near-perfect move ordering, and which gives the engine an
always-ready move for the clock. You understand the horizon effect — why a
fixed-depth search is fooled by pending captures — and quiescence search as the
fix: a restricted search through captures (and checks) with a stand-pat floor and
delta/SEE pruning, so that `evaluate` is only ever called on quiet positions. You
know aspiration windows as a narrow root window centered on a guess, cheap when
right and re-searched wider when wrong, and you understand search instability as
the natural consequence of comparing windowed, bounded search results across
iterations — a feature of the machinery, not a defect.

**Next:** `07-the-transposition-table.md` — the same position is reached by many
different move orders (the game tree is really a graph), and re-searching it is
pure waste. The transposition table caches search results in a giant hash table,
but doing so under a fixed memory budget, across many threads, while respecting the
bound-type subtleties of file 05, is a genuinely hard systems problem.
