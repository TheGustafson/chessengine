# 10 — Pruning and reductions: spending nodes where they matter

File 09 taught the search *what to look at first*. This file is about the
opposite and equally vital skill: *what not to look at at all*, and *what
to look at only shallowly*. These are the pruning, reduction, and
extension heuristics — the layer that turns a correct-but-slow alpha-beta
search into a Stockfish-class one. In ARCHON they arrived in "waves" that
were worth, cumulatively, something like **+570 Elo** and then another
**+237 Elo** on top. That is the difference between a decent club player
and a strong one, bought almost entirely by deciding which branches to
ignore.

The owning design document is `design/10-pruning-extensions.md`, which
specifies the exact order of operations at every search node. We will not
reproduce all of it; we will build the *ideas* from first principles and
point at ARCHON's real formulas where they illuminate.

## 1. The error-budget mental model

Every pruning decision is a **bet**. When the search says "I will not
examine this move" or "I will examine it only to shallow depth," it is
wagering that the move is not the best one. Usually it wins the bet and
saves a huge subtree of work. Occasionally it loses — the pruned move was
actually best — and the engine returns a slightly wrong score, which can
cost it the game.

So think of it as a budget. Each heuristic has:

- an **upside**: the nodes it saves, which translate into extra search
  depth elsewhere, which translates into Elo; and
- a **cost**: the probability it is wrong times the damage when it is
  wrong.

A pruning rule is worth adding **only if its expected upside exceeds its
expected cost.** This is not a metaphor — it is literally how these rules
are validated. You implement the heuristic, play tens of thousands of
games with and without it (file 14 explains the statistics), and keep it
only if it measurably wins. The formulas and their exact constants are
*tuned* to sit right at the edge where saving one more node starts to cost
more Elo than it gains.

Two consequences shape everything below:

1. **Prune more where you are confident, less where you are not.** The
   whole art is estimating confidence cheaply. Most of the heuristics are
   really just different cheap proxies for "how sure am I that this branch
   is irrelevant?"

2. **Never prune when you might be getting mated.** A wrong prune in a
   quiet position costs a fraction of a pawn. A wrong prune that misses a
   forced mate loses the game outright. So every pruning rule in ARCHON
   carries guards like `!is_loss(best_value)` — do not prune while we
   still might be losing to force — and refuses to fire in check. The
   asymmetry of the downside is baked into the guards.

Keep this budget framing in mind. Each rule below is one line item in it.

## 2. Two directions: pruning and reducing

There are two ways to spend fewer nodes on a move:

- **Pruning** — skip the move entirely (or return early from the whole
  node). Zero nodes spent.
- **Reducing** — search the move, but to *less depth* than normal. Some
  nodes spent, but far fewer than a full-depth search.

Pruning is the aggressive bet; reducing is the hedge. Much of ARCHON's
strength comes from a clever hedge (late move reductions, §5) that is
*cheap insurance*: reduce first, and only pay for a full search if the
reduced one surprises you. We will build up from the pure pruning rules to
that hedge.

## 3. Reverse futility pruning (the "we're already winning" cut)

Start with the simplest confident bet. Suppose it is our move, our static
evaluation of the position is already *far above* `beta` — the score the
opponent has proven they can hold us to — and the position is quiet. Then
it is overwhelmingly likely that *some* move keeps us above `beta`, so the
node will fail high no matter what. Why search at all?

**Reverse futility pruning** (RFP, also called "static null move
pruning") makes exactly this bet: if `eval − margin ≥ beta`, return
immediately without searching any move. The `margin` grows with depth
(deeper nodes demand more cushion, because there is more time for things
to go wrong) and shrinks when we are "improving." ARCHON's version, from
`design/10`:

```
   margin = futility_mult * depth − (improving/worsening adjustments)
                                  + |eval uncertainty| / big_divisor
   if eval − margin ≥ beta:  return a value blended toward beta
```

Two details reveal the engineering. First, it does not return the raw
`eval`; it returns a **blend toward `beta`**, damping the noise in an
optimistic static evaluation. Second, the margin *widens* when the
correction-history signal from file 09 says "the eval is unreliable
here" — the engine prunes less confidently exactly where it has learned
its own eval tends to be wrong. That coupling between a learned signal and
a pruning margin is a small but telling piece of craft.

RFP only runs at shallow-to-medium depth (`depth < 17` in ARCHON) and
never when a mate is in play. It is a high-frequency, low-risk cut — a big
line item in the budget.

## 4. Null-move pruning, zugzwang, and verification

This one is beautiful, and it requires an idea that sounds illegal:
**letting a side skip its turn.**

### 4.1 The intuition

In almost every chess position, having the move is an *advantage* — you
can always find *something* useful to do. So here is a bet. Suppose we let
the opponent move twice in a row (we "pass" — make a **null move**) and
then search the resulting position to reduced depth. If even *after
generously handing the opponent a free move* our position is still good
enough to beat `beta`, then our real position — where we *don't* skip our
turn — is surely at least that good. We can fail high immediately.

That is **null-move pruning**. Concretely: make a null move (flip the side
to move, do nothing else), search to depth `depth − R` for some reduction
`R`, and if that "give the opponent a free move" search still returns a
score `≥ beta`, prune the whole node. ARCHON uses `R = 7 + depth/3` — a
large reduction, because the null-move search is only a cheap sanity check,
not a real line.

The payoff is enormous. Null-move pruning is one of the highest-value
heuristics in all of computer chess, because it lets the engine dismiss
huge swaths of "we're comfortably ahead" positions after a single shallow
probe.

### 4.2 Zugzwang: where the bet fails

The bet rests on "having the move is an advantage." There is a class of
positions where that is **false**: **zugzwang** — from the German for
"compulsion to move" — positions where every legal move makes things
worse, and the side to move would love to pass but is not allowed to.
These occur mostly in endgames with few pieces. In a zugzwang position,
the null-move assumption is exactly backwards: passing is *better* than
any real move, so a null-move search over-estimates the position and the
prune can be catastrophically wrong.

ARCHON guards against this two ways. First, it refuses null-move pruning
when the side to move has **no non-pawn material** (`non_pawn_material(us)
!= 0` is required) — pawn-and-king endgames are where zugzwang lives, so
it simply does not make the bet there. Second, and more generally, it uses
**verification**.

### 4.3 Verification search

At higher depths, when a null-move prune *would* fire, ARCHON does not
immediately trust it. It runs a **verification search**: a real search (no
null move) at the reduced depth, with null-move pruning *disabled* for the
rest of that subtree. Only if the verification also says "`≥ beta`" does
the prune stand. The mechanism that disables null moves for the subtree is
a per-worker `nmp_min_ply` marker — a ply threshold below which null moves
are forbidden — which is set before the verification and reset after. This
makes the scheme recursion-proof and zugzwang-proof: a position that only
*looked* winning because of a lucky null move gets caught by the
verification that is not allowed to cheat the same way.

Verification is the budget framing in action: at low depth the wrong-prune
cost is small, so ARCHON skips verification and takes the cheap prune; at
high depth the cost is large, so it pays for the insurance.

## 5. Late move reductions: cheap insurance

If null-move pruning is the boldest bet, **late move reductions** (LMR)
are the most important *hedge* — and probably the single most valuable
search technique of the modern era.

### 5.1 The idea

Recall from file 09 that moves are searched in order of how likely they
are to be best. That means: by the time you reach the *tenth* move at a
node, ordering is telling you this move is probably mediocre. Searching it
to full depth, like the first move, is almost certainly wasted effort — but
you cannot *skip* it, because "probably mediocre" is not "certainly
mediocre."

LMR threads this needle. Search the late move to **reduced depth** first.
If — as expected — it comes back below `alpha` (confirming it is
mediocre), you are done, cheaply. But if the reduced search *surprises*
you by beating `alpha`, you do not trust the shallow result: you
**re-search at full depth** to get the real answer. This is
"reduce-then-research," and it is cheap insurance because the expensive
full re-search only happens on the rare move that surprised you.

```
   reduced search at (new_depth − r):
       result ≤ alpha  →  believe it, move on            (the common case)
       result >  alpha  →  re-search at full new_depth    (the rare case)
```

The savings are massive: the great majority of late moves get a shallow
search and nothing more, while the occasional good late move is still
found because the re-search catches it.

### 5.2 The log-log reduction table

How much to reduce — the value `r` — is the crux. The insight, standard
since the mid-2010s, is that the right reduction grows roughly with the
**logarithm** of both the depth and the move number. Reduce more when
you are deep (there is depth to spare) and more for later moves (ordering
trusts them less), but with *diminishing* increments — the difference
between move 20 and move 30 matters less than between move 2 and move 12.

ARCHON precomputes a small table once at worker startup:

```
   reductions[i] = (2834 / 128) * ln(i)          for i in 1..256
```

and the per-move base reduction multiplies the depth entry by the
move-number entry: `reductions[depth] * reductions[move_number]`. That
product of two logs is the "log-log" shape. It is then adjusted by a
whole list of situational nudges — reduce less on the principal variation,
reduce less when the move has good history (file 09's tables feed straight
in here), reduce more in "cut nodes" where a fail-high is expected, and so
on. The base table is computed with real floating-point `ln` because it
runs exactly once, before any search, so its cost is irrelevant.

Everything is carried in **1024ths of a ply** and divided down only at the
moment of application — integer-only arithmetic in the hot path, the log
computed once at setup. That is a recurring pattern: do the expensive,
floating-point, or irregular work once at initialization, and keep the
per-node path to cheap integer adds and shifts.

### 5.3 Deeper and shallower

There is a lovely refinement. After a reduced search that beats `alpha`
and triggers a re-search, ARCHON asks whether the move was *so* good it
deserves *more* than full depth, or the surprise was marginal enough to
search *less*. It nudges `new_depth` up or down by comparing the result to
`best_value` with small margins, then re-searches at the adjusted depth.
The reduction is not a fixed penalty; it is a continuously adjusted
estimate that keeps getting corrected by what the search actually finds.

## 6. The shallow-depth pruning family

Near the leaves of the tree — where `lmr_depth` (the *predicted* depth
after reductions) is small — a cluster of cheaper, more aggressive rules
kicks in. Each is a different cheap proxy for "this move can't matter."

**Late move pruning (LMP), a.k.a. movecount pruning.** Once you have
already tried *enough* moves at a node without any of them cutting off,
just stop generating quiet moves. The threshold grows quadratically with
depth: `move_count ≥ (3 + depth²) / (2 − improving)`. The bet: if the
first `k` well-ordered moves all failed to raise `alpha`, move `k+1`
(which ordering ranked even lower) almost certainly will not either. In
ARCHON this literally tells the move picker to `skip_quiet_moves()` — the
remaining quiets are *never even generated*, tying straight back to the
lazy staging of file 09.

**Futility pruning (parent).** For a quiet move near the leaves: if the
static eval plus a depth-scaled margin is *still below `alpha`*, the move
cannot plausibly raise `alpha`, so skip it. The picture: "even if this
quiet move gains a little, I'm so far below `alpha` that it can't close
the gap." The margin includes small bonuses for "no best move found yet"
and "eval already above alpha," each a nudge in the confidence estimate.

**Capture futility.** The same idea for captures, but the margin must
account for the material the capture *wins* — you cannot dismiss a capture
without crediting it the victim's value. So: if `static_eval + margin +
piece_value(captured) + capture_history_term ≤ alpha`, skip it. Only fires
when the capture does not give check and the predicted depth is shallow.

**SEE pruning (capture and quiet).** File 09's static exchange evaluation
returns here as a *pruning* gate. For captures: skip if the capture loses
more than a depth-scaled margin of material (`!see_ge(m, −margin)`). For
quiets: skip if the move loses material to an immediate recapture, with a
margin that grows *quadratically* with the predicted depth
(`−25 * lmr_depth²`). A quiet move that walks a piece onto a square where
it is simply lost is pruned before wasting a search on it — unless
sacrificing the last piece might be a stalemate trick, which the guards
explicitly protect.

**Razoring.** The most aggressive of the family. If, at a non-principal
node, the static eval is *so* far below `alpha` (`eval < alpha − 465 −
300*depth²`) that no reasonable sequence of quiet moves could rescue it,
skip the full search and just run **quiescence search** (file 06) to check
for tactics. The quadratic margin is huge, which makes razoring
self-limiting — it only fires when the position is hopeless-looking — so it
needs no separate depth cap.

**Continuation-history pruning.** A pure history bet: if a quiet move's
combined continuation-history score is very negative (`history <
−4313*depth`), the engine has *learned within this search* that moves like
this keep failing here, so it prunes. This is file 09's learning feeding
directly into file 10's pruning.

Every one of these shares the master guards — never in check, never when
being mated, only near the leaves — and every one is a separate line in
the error budget, individually validated by games.

## 7. Extensions: spending *more* where it matters

Pruning and reducing spend *fewer* nodes on unpromising moves. The
complement is spending *more* on the rare move that deserves it. The main
mechanism in ARCHON is the **singular extension**, and it is subtle.

### 7.1 Singular extensions

Suppose the transposition table strongly suggests one move (the "TT move")
is best here. If it is best by a *large margin* — if *every other move* is
clearly worse — then this move is "singular": the position really hinges on
it, and it is worth searching **one ply deeper** to make sure we understand
it. Missing the point in a position with one critical move is exactly how
engines walk into tactics.

But how do you prove a move is singular without searching all its rivals
to full depth? The trick: run a **reduced-depth search of the node with
the TT move excluded**, using a window placed just *below* the TT move's
known value. If that "everyone except the TT move" search comes back
*below* the window — meaning no other move can even come close — the TT
move is singular, and the node's depth is bumped by one (sometimes two or
three, for extreme margins). ARCHON implements exactly this: it sets an
`excluded_move`, searches at `singular_depth = new_depth / 2` with a
`singular_beta` derived from the TT value, and reads the outcome.

Note the elegance: to decide whether to spend *more* on one move, the
engine spends a *little* on a cheap probe of its neighborhood. The probe is
half-depth and single-windowed, so it is far cheaper than actually
searching every rival to full depth.

### 7.2 Multicut: the same probe, the opposite conclusion

The singular probe can also *fail* in an informative way. Suppose the
"everyone except the TT move" search comes back *above* `beta` — meaning
**several** moves here beat `beta`, not just the TT move. Then the node is
the opposite of singular: it is a position where lots of moves are good
enough to cause a cutoff. If multiple moves already beat `beta` in the
reduced probe, the full node will surely fail high, so ARCHON prunes the
whole subtree immediately and returns. This is **multicut** — one probe,
and depending on which way it breaks, you either extend (singular) or prune
the entire node (multicut). Two of the highest-leverage decisions in the
search fall out of the same cheap exclusion search.

There are also **negative extensions**: if the TT move is *expected* to
fail high anyway, search it slightly *shallower* to spend the saved depth
on its alternatives. Extensions in ARCHON are entirely of this data-driven,
singular/multicut kind — notably, there are **no plain "extend on check"
extensions**, a deliberate choice to follow current Stockfish practice over
older folklore.

## 8. Why they compound multiplicatively

Here is the payoff, and the reason ARCHON's pruning arrived as "+570 then
+237 Elo" rather than a modest sum. These heuristics do not *add* their
savings; they **multiply** them.

Think about it structurally. Null-move pruning removes, say, a third of
the nodes. On the two-thirds that remain, LMR reduces the depth of most
late moves, shrinking each surviving subtree. On what remains of *those*,
razoring and futility lop off the shallow tails. On what remains of *that*,
LMP stops generating moves entirely. Each rule operates on the tree *the
previous rules already shrank*. If rule A leaves a fraction `f_A` of the
tree and rule B independently leaves `f_B`, together they leave roughly
`f_A * f_B`. Stack six or eight such rules and the tree collapses by two or
three orders of magnitude — which is exactly what lets a modern engine
reach depth 25 where a naive alpha-beta reaches depth 10 in the same time.

That multiplicative structure is also why the rules are tuned and tested
*together* and in a specific *order*. The order at a node is not arbitrary:
`design/10` fixes it precisely (mate-distance pruning, then TT cutoff, then
RFP, then null move, then the move-loop family, then singular extensions),
because each rule assumes the ones before it have already run and cheapened
the state it reads. Reordering them, or removing one from the middle,
changes what the others see and can quietly break the whole cascade. The
engine's design doc treats the node walkthrough as a single, carefully
sequenced pipeline for exactly this reason.

And it is why the "error budget" framing matters so much. When every rule
is riding on the savings of every other rule, a single miscalibrated
margin does not just cost its own line item — it can poison the confidence
estimates that later rules depend on. The reason ARCHON's constants are
transcribed from a world-class engine and then re-tuned on its own games
(file 14) is that this cascade only pays off when every stage in it is
sitting right at the edge of its budget.

## What you now know

- Every prune or reduction is a **bet**: Elo gained from nodes saved,
  minus the cost of being wrong, with an asymmetric downside that makes
  "never prune when you might be mated" a hard rule.
- **Reverse futility pruning** cuts nodes that are already comfortably
  above `beta`; **null-move pruning** dismisses winning positions by
  letting the opponent move twice, guarded against **zugzwang** by a
  no-non-pawn-material rule and a **verification search**.
- **Late move reductions** are cheap insurance — search late moves
  shallow, re-search only on a surprise — with reduction growing as the
  **log of depth times the log of move number**.
- The shallow-depth family (LMP, parent/capture futility, SEE pruning,
  razoring, continuation-history pruning) are cheap proxies for "this move
  can't matter," each individually game-validated.
- **Singular extensions** and **multicut** both fall out of one cheap
  "exclude the TT move and probe its neighborhood" search — extend when
  one move is uniquely critical, prune the whole node when many moves are.
- These heuristics **compound multiplicatively**, which is why they must
  be tuned and ordered together, and why they were worth hundreds of Elo.

We have now built a complete classical engine: representation, move
generation, state, alpha-beta with iterative deepening and quiescence, a
transposition table, ordering, and this pruning cascade — all riding on a
hand-crafted evaluation. The next file changes the last piece. Instead of
hand-writing the evaluation, we *learn* it with a neural network small and
fast enough to run at every one of those millions of nodes. Continue to
`11-nnue-how-a-network-evaluates-fast.md`.
