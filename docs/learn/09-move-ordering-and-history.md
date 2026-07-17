# 09 — Move ordering and history: teaching the search where to look

In file 05 we met alpha-beta, and we met the single most important fact
about it: **its speed depends entirely on the order in which you try
moves.** This file is about how ARCHON exploits that fact. It is, in a
real sense, where a chess engine earns most of its strength — more than
from evaluating positions accurately. By the end you will understand
MVV-LVA, static exchange evaluation (SEE), the staged move picker, and
the family of "history" tables that let the engine *learn*, within a
single search, which moves tend to be good.

## 1. Why ordering beats evaluation at fixed depth

Recall the punch line from file 05. Alpha-beta searches a tree of
positions. At every node it tries moves one at a time; the moment a move
proves "good enough to refute this line" (it raises the score to `beta`
or above), the node **cuts off** — the remaining moves are never
examined. If you always tried the refuting move *first*, you would
examine exactly one move per node before cutting. If you always tried it
*last*, you would examine all of them.

The theory makes this quantitative. For a tree of branching factor `b`
and depth `d`, plain minimax visits about `b^d` leaves. Alpha-beta with
**perfectly** ordered moves visits about `b^(d/2)` — the *square root*.
That is the difference between searching to depth 8 and depth 16 in the
same time. With **random** ordering you get almost none of the benefit.
Real engines land close to the perfect-ordering ideal, and the gap
between "close" and "not close" is enormous.

Here is the part people find surprising the first time. Suppose you have
two engines that search to the *same fixed depth*. Engine A has a
world-class evaluation function but tries moves in a mediocre order.
Engine B has a crude evaluation but orders moves brilliantly. At fixed
depth, **B is usually the stronger searcher** — not because it evaluates
better, but because good ordering makes alpha-beta prune so aggressively
that B can afford to search *deeper* in the same wall-clock time, and
depth dominates. Ordering is a force multiplier on everything else.

This is why an engine invests so much machinery in a question that
sounds trivial — "which move should I try next?" — and why this file is
longer and denser than you might expect for a topic that is, formally,
just sorting.

The goal of move ordering is therefore: **try the move most likely to
cause a cutoff first.** We do not know which move that is (if we did, we
would not need to search). So we *estimate* it, using several sources of
evidence, cheapest and most reliable first.

## 2. The cheapest evidence: captures and MVV-LVA

The single best predictor of "this move is strong" is: **it captures a
valuable piece.** Winning a queen is good in almost every position, so a
queen capture is an excellent first guess for the refuting move.

The classic heuristic is **MVV-LVA**: *Most Valuable Victim, Least
Valuable Attacker*. Order captures so that:

- you grab the most valuable **victim** first (queen before rook before
  bishop before pawn), and
- among captures of the same victim, you use the **least valuable
  attacker** first (capture the queen with a pawn before capturing it
  with a rook).

The intuition for the second half: if you can win the queen with a pawn,
that is almost certainly better than winning it with your own rook,
because you risk less. A tiny worked example. Black has a queen on d5.
White can play `pawn takes d5` or `rook takes d5`. Both win the queen,
but MVV-LVA tries the pawn capture first: same reward, less exposure, so
it is the more likely cutoff.

ARCHON's capture scorer (`src/movepick.rs`, `score_capture`) is a
slightly modernized MVV-LVA. It scores a capture as

```
CAPTURE_MVV_FACTOR * victim_value + capture_history[mover][to][victim]
```

Notice two things. First, there is **no explicit attacker term** — the
"least valuable attacker" idea is handled later, by the good/bad split in
§4, which asks a sharper question than piece values alone. Second, there
is a `capture_history` term added on: a *learned* correction we will
reach in §6. The base signal is still "grab the big piece first."

Victim values here are small integers (pawn ≈ 1, knight/bishop ≈ 3,
rook ≈ 5, queen ≈ 9, in the engine's internal units). But a raw victim
value is a blunt instrument: it says "capturing the queen is great"
without checking whether the queen is *defended*. Capturing a defended
queen with your rook, when the recapture wins your rook back and more,
is a blunder. To rank captures honestly we need to simulate the whole
exchange. That is what SEE does.

## 3. Static exchange evaluation (SEE) from first principles

**Static exchange evaluation** answers one question about one square: *if
both sides keep capturing on this square, trading their cheapest
available attacker each time, what is the net material result?* "Static"
means we do not search a tree of real moves — we only shuffle attackers
on and off a single square, using cheap piece values. It is a
mini-simulation, not a real search, which is why it is fast enough to run
millions of times per second.

### 3.1 The swap-off, by hand

Picture a square — call it e5 — with a black pawn sitting on it. White
wants to know: is capturing on e5 good? The attackers and defenders of
e5 are:

```
        White attackers of e5 : knight (3), rook (5)
        Black defenders of e5 : pawn (1), queen (9)
        Piece currently on e5 : black pawn (value 1)
```

SEE plays out the forced exchange, **always using the cheapest attacker
available to the side to move**:

1. White's cheapest attacker is the knight (3). White plays
   knight-takes-pawn. White has won a pawn: running gain to White = **+1**.
   The knight now sits on e5.
2. Black's cheapest defender is the pawn (1). Black plays
   pawn-takes-knight. Black wins the knight: from White's view the
   running total is `+1 − 3 = −2`. A black pawn sits on e5.
3. White's cheapest remaining attacker is the rook (5). Rook-takes-pawn:
   `−2 + 1 = −1`. Rook on e5.
4. Black's cheapest remaining defender is the queen (9).
   Queen-takes-rook: `−1 − 5 = −6`. Queen on e5.
5. White has no attackers left. The exchange ends.

Now walk the running totals **backward**, because each side, at its turn,
will only continue the exchange if doing so improves its result — nobody
is forced to recapture into a loss. The sequence of White-relative
material after each capture was `+1, −2, −1, −6`. Each side, on its move,
takes the *better* of "stop now" versus "capture and let the opponent
choose". Folding from the leaf back to the root with that rule yields the
value White can actually guarantee: here White should stop after winning
the first pawn, because continuing walks into `−2`. The SEE value of
capturing on e5 is **+1** — a clean won pawn.

That backward fold is the crux. The forward pass generates the candidate
material swings; the backward pass applies the fact that either side can
bail out the instant continuing hurts them. In code it is a small loop
over an array, no recursion.

### 3.2 What engines actually ask

In practice you rarely want the exact SEE *number*; you want a
**threshold** answer: "does this capture win at least `T` material?" That
is the `see_ge(move, threshold)` predicate (in `src/board.rs`).
`see_ge(m, 0)` asks "does this capture not lose material?" — the standard
"is this a good capture?" test. `see_ge(m, -50)` asks "does it lose at
most half a pawn?" — a looser gate used when we are willing to tolerate a
small material investment for other reasons.

Thresholded SEE is cheaper than full SEE because you can stop the swap
loop the moment the answer is decided one way or the other. ARCHON uses
it in two very different places: **move ordering** (this file, to split
good captures from bad) and **pruning** (file 10, to throw away captures
and quiet moves that lose too much material to be worth searching). Same
mini-simulation, two consumers.

A caution SEE embodies: it is *static*. It ignores pins, overloaded
defenders, in-between moves (zwischenzug), and checks. A defender that is
pinned to its king cannot actually recapture, but plain SEE counts it
anyway. This is a deliberate trade: SEE is an ordering and pruning
*heuristic*, fast and usually right, and the real search corrects it when
it matters.

## 4. The staged move picker: don't sort what you won't look at

Now we have scores for captures. The naive plan would be: generate *all*
legal moves, score every one, sort the whole list, and hand them to the
search in order. This is wasteful, and understanding *why* is the key
idea of this section.

Most nodes cut off almost immediately — after the transposition-table
move, or after the first good capture. If a node is going to cut off on
move 1, then generating and scoring its other 30 moves was pure waste.
Sorting them was worse than waste. So ARCHON's `MovePicker`
(`src/movepick.rs`) is **lazy** and **staged**: it produces one move at a
time, and it does the minimum work needed to produce the *next* move,
deferring everything else until the search actually asks for it.

The stages, in order, for a normal search node:

```
  1. TT move        — the move the transposition table suggests (file 07)
  2. Good captures  — captures that pass their SEE gate, best first
  3. Quiets         — non-captures, ordered by history (§5)
  4. Bad captures   — captures that FAIL the SEE gate (deferred to here)
  5. Bad quiets     — quiets whose history score fell below a threshold
```

Why this exact order? Each stage is a bet that its moves are more likely
to cut off than the next stage's. The **TT move** comes first because it
was the best move found in this position on a previous, possibly
shallower search — a very strong prior. **Good captures** come next: a
capture that SEE says wins material is a natural refutation candidate.
**Quiets** are the vast middle, ordered by learned history. **Bad
captures** — captures that lose material by SEE — are searched *after*
the quiets, because a losing capture is usually a mistake, but occasionally
it is a brilliant sacrifice, so you cannot discard it entirely; you just
try it late.

The staging pays off precisely because of stage laziness. If the TT move
causes a cutoff, the picker has generated *nothing else* — zero capture
generation, zero quiet generation, zero sorting. ARCHON's picker makes
this explicit: on a TT-move cutoff it "performs zero generation, scoring,
and sorting," as its own module docs put it. That is the whole point of
staging over sorting.

### 4.1 The good/bad capture split

How does the picker decide a capture is "good"? When it reaches the
captures, it generates and scores them, then emits them one at a time in
descending score order. For each candidate it runs `see_ge` with a
threshold that *scales with the ordering score*: a capture with a high
MVV-plus-history score is allowed to fail SEE by a bit and still count as
good (the ordering score bought it some slack), while a capture with a
mediocre score must pass SEE cleanly. Captures that fail are **swapped
down** into a "bad captures" region of the buffer, physically moved out
of the way, and revisited only at stage 4.

One implementation detail worth noticing, because it is a nice example of
the engine's "no waste" philosophy. The picker owns a single fixed-size
array (`[ExtMove; 256]`, an `ExtMove` being a move plus its `i32` score)
and carves it into regions — bad captures, captures, quiets — using
integer cursors rather than allocating separate lists. Everything happens
in place, with no heap allocation on the hot path, which matters when you
construct a fresh picker at every one of millions of nodes.

### 4.2 Partial sorting

Even within a stage, ARCHON does not fully sort. It uses a **partial
insertion sort** with a score `limit`: only moves scoring above the limit
get sorted into place; the rest are left in a jumbled tail. The limit for
quiets scales with depth, so deep nodes (where getting the order right
really matters) sort a large prefix, while shallow nodes barely sort at
all. Sorting effort tracks how much of the list the search will plausibly
consume. Fully sorting a 40-move quiet list at shallow depth would be, in
the module's own words, "pure waste."

## 5. History heuristics: reinforcement learning in miniature

Captures we can order with SEE. But most moves in most positions are
**quiet** — they capture nothing. How do you order 30 quiet moves, none
of which grabs material? You cannot use MVV-LVA (no victim) and SEE says
"zero" for all of them.

The answer is one of the most elegant ideas in computer chess:
**let the engine learn, as it searches, which quiet moves tend to be
good — and reuse that knowledge across the millions of nodes in one
search.** This is a small, self-contained instance of reinforcement
learning. There is no neural network here; just a big table of integer
counters that get rewarded when a move works and punished when it does
not.

### 5.1 The core learning rule: gravity

Every history table in ARCHON is built from one primitive, a single
counter cell that updates with a **"gravity" rule**. The cell holds an
`i16` bounded to `[−D, +D]` for some table-specific limit `D`. When a
move earns a bonus `b`, the cell updates as:

```
   new = old + clamp(b, −D, D) − old * |clamp(b, −D, D)| / D
```

Stare at that for a second, because the second term is the clever part.
It says: move toward the bonus, but *decay the old value in proportion to
how big the bonus is relative to the ceiling `D`*. Three consequences,
all of which the engine's unit tests actually verify:

- The cell **never escapes** `[−D, D]`. The subtraction always pulls back
  just enough.
- Near the ceiling, bonuses have **diminishing effect**. A cell already
  near `+D` barely moves when rewarded again — it is "saturated." This is
  the "gravity" metaphor: the further from zero, the harder it is to go
  further.
- When the signal flips or stops, the old value **decays geometrically**
  toward zero. Stale knowledge fades.

This is exactly the behavior you want from an online learner with bounded
memory: recent, consistent evidence dominates; a single fluke barely
moves the needle; and old lessons are forgotten gracefully. In ARCHON
this lives in `src/history.rs` as a type `StatEntry<const D: i32>`. The
`const D` is a *const generic* — a compile-time integer parameter that
lets one type carry a different saturation ceiling per table (the
butterfly table uses `D = 7183`, continuation history `D = 30000`, and so
on) with the compiler generating specialized arithmetic for each. It is
Rust's idiomatic replacement for a family of near-identical structs or a
macro.

### 5.2 Butterfly history: the base signal

The simplest table is **butterfly history**, indexed by `(side to move,
from-square, to-square)`. When a quiet move causes a cutoff, its
butterfly cell gets a positive bonus; when a quiet move is tried and
*fails* to cut off (while some later move succeeds), it gets a penalty.
Over a whole search, cells for genuinely strong quiet moves drift toward
`+D`, and the quiet scorer reads these values to order quiets.

The name is historical: the `(from, to)` pairs form a "butterfly board."
The bonus size scales with depth — a cutoff proven at depth 12 is much
stronger evidence than one at depth 2 — and the penalties for the tried-
but-failed moves *decay geometrically* down the list, so the moves tried
earliest (and thus most trusted by ordering) are punished hardest when
they turn out wrong. That asymmetric reward/punishment is the "learning"
signal.

### 5.3 Capture history: even captures get a memory

We saw in §2 that the capture scorer adds a `capture_history` term.
Captures get their own history table too, indexed by `(moving piece,
to-square, victim type)`. This corrects MVV-LVA with experience: if
"rook takes on f7" keeps failing in this search even though it wins a
pawn by raw value, its capture-history cell goes negative and it sinks in
the ordering. Learned nuance on top of the static victim value.

### 5.4 Continuation history: context matters

Here is the subtle, powerful one. A quiet move's strength often depends
on **what just happened**. Knight-to-e5 might be excellent as a reply to
the opponent's bishop-to-g4, and mediocre otherwise. Plain butterfly
history cannot express this — it only knows `(from, to)`, not the
*context* of the preceding moves.

**Continuation history** indexes by the *predecessor move* as well: it
scores "playing piece P to square T, *given that the previous move was
piece Q to square R*." So the engine learns move **pairs** and
**sequences**, not just isolated moves. ARCHON actually maintains several
continuation planes at once — reacting to the move 1 ply ago, 2 plies
ago, up to 6 plies ago — and the quiet scorer sums the relevant ones. The
1-ply and 2-ply planes are the famous "counter-move" and
"follow-up-move" histories under a unified roof.

The context is what makes this work. "Develop the knight" is not
universally good or bad; "develop the knight *in response to this pawn
push*" is a much sharper, more predictive statement, and continuation
history is how the engine accumulates statements at that resolution.

There is a genuinely interesting Rust wrinkle here. Stockfish stores a
raw pointer to the relevant continuation slab on its search stack. ARCHON
cannot — a raw pointer would need `unsafe`, and a normal borrow would
freeze the whole multi-megabyte table while the search still needs to
mutate other parts of it. Its solution is a small `Copy` integer handle,
`ContHistId`, that flattens `(in_check, capture, piece, to)` into a
single index in `0..3328`. The handle sits inertly on the search stack;
when the scorer needs the slab it resolves the index in one array load.
The move is: **replace an aliasing pointer with a plain index**, which is
the standard Rust technique for keeping a pointer-heavy data structure
both `unsafe`-free and acceptable to the borrow checker.

### 5.5 Pawn and low-ply history

Two more small tables round out the quiet scorer. **Pawn history** is
indexed by a hash of the pawn structure plus `(piece, to)`, capturing the
fact that the same piece move can be good or bad depending on the pawn
skeleton around it. **Low-ply history** is a short-lived table for the
first few plies of the search, refilled each new search, that helps order
moves near the root where getting it right is most valuable. Each is a
`StatEntry` table with its own ceiling `D`, updated by the same gravity
rule.

## 6. Correction history: learning the *eval's own error*

We finish with ARCHON's newest search feature — and one of its biggest
recent wins. Wiring it into the search was worth about **+73 Elo** at
equal time, the largest single search gain since the big pruning waves.

The idea generalizes history from "which *moves* are good" to "how wrong
is my *evaluation* here, and in which direction?" When the engine
statically evaluates a position it gets a number — say +40 centipawns.
Then it searches, and the search reveals the *true* value was more like
−20. The difference, `searched − static = −60`, is the static eval's
**error** at this position. That error is not random noise: it correlates
with structural features of the position. Some kinds of positions the
static eval systematically over-rates; others it under-rates.

**Correction history** learns those systematic errors online and folds a
correction back into the static eval before it is used for pruning
decisions. The engine keys the error signal several ways — by pawn
structure, by minor-piece configuration, by non-pawn material per side,
and by move context (a continuation-style key) — so it learns "positions
with *this* pawn skeleton tend to be worth `X` less than the raw eval
says." Each key indexes its own `StatEntry` table (ceiling `D = 1024`),
updated by the same gravity rule you now know well.

At evaluation time, ARCHON reads all the correction keys, combines them
into a weighted correction, and produces a *corrected* static eval
(`history.rs`, `correction_value` and `to_corrected_static_eval`).
Crucially, the **raw** eval is still what gets stored in the
transposition table; the **corrected** eval is what drives the
pruning-and-reduction decisions of file 10. The update is carefully gated
— it only fires when the error signal is trustworthy (not in check, when
the best move was not a capture, and only when the direction of the error
agrees with whether a good move was actually found), so the engine does
not teach itself from noise.

It is worth appreciating the unity here. Butterfly, capture, continuation,
pawn, and correction history are *all* the same machine: bounded integer
cells, updated by one gravity rule, rewarded by search outcomes, decaying
when stale. The engine is running five small reinforcement learners in
parallel, each answering a slightly different question — which quiet is
good, which capture is good, which move-pair is good, which move fits
this pawn structure, and how wrong is my eval — and all of them feed the
one decision that dominates search efficiency: **what to look at next.**

## What you now know

- At **fixed depth**, move ordering matters more than evaluation quality,
  because good ordering is what turns alpha-beta's `b^d` into `b^(d/2)`
  and buys extra depth.
- **MVV-LVA** orders captures by victim value (and, implicitly, attacker
  risk); **SEE** refines this by simulating the full swap-off on one
  square with a cheap forward-then-backward fold, exposed as the
  thresholded `see_ge` predicate.
- The **staged, lazy MovePicker** produces moves one at a time — TT move,
  good captures, quiets, bad captures — so a node that cuts off early
  pays almost nothing, and it partial-sorts only what the search will
  plausibly consume.
- **History tables** are a miniature reinforcement-learning system: a
  single bounded "gravity" update rule powers butterfly, capture,
  **continuation** (context-aware, indexed by predecessor moves), and
  pawn history, all rewarded and punished by search outcomes.
- **Correction history**, ARCHON's newest and one of its most valuable
  search features (+73 Elo), learns the *static eval's own systematic
  error* online and corrects it before pruning.

Ordering tells the search *what to look at first*. The next file is about
the flip side — **what to skip entirely**: the pruning, reduction, and
extension heuristics that let the tree collapse from billions of nodes to
millions, and the "error budget" way of thinking about every one of them.
Continue to `10-pruning-and-reductions.md`.
