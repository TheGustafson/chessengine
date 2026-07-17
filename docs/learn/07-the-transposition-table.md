# 07 — The transposition table

The game tree is not really a tree. Reach a position by 1.e4 e5 2.Nf3, or by
1.Nf3 e5 2.e4, and you arrive at the *same* position by two different paths.
These convergences are **transpositions**, and they mean the "tree" is actually a
directed acyclic graph (DAG) with the same node appearing in many places. Search,
which walks it as if it were a tree, will re-examine the same position over and
over — pure waste, and a *lot* of it. The **transposition table** (TT) is the
cache that fixes this: a giant hash table remembering the result of searching each
position, so a repeat visit is a lookup instead of a re-search. Getting it right is
one of the harder systems problems in an engine, because it must live in a fixed
memory budget, tolerate hash collisions, respect the bound-type subtleties of file
05, and be read and written by many threads at once without locks. This file works
through all of that. Ground truth is `src/tt.rs`.

## Why a cache helps so much

Consider how often positions transpose. In a middlegame, moves can frequently be
played in different orders to reach the same place; in the endgame, with few pieces,
transpositions are pervasive. Empirically, adding a transposition table to an
otherwise-identical search cuts the node count dramatically — ARCHON measured about
a 47% reduction at equal depth when its TT landed. Nearly half the search was
redundant. And it is not only about avoiding re-searches: even when a TT entry is
not deep enough to reuse its *value*, it usually remembers the *best move* found
last time, which is a superb move-ordering hint (file 05: ordering is everything).
The TT pays off twice — as a result cache and as an ordering oracle.

The lookup key is the Zobrist hash from file 04. Two positions that are truly
identical have the same key; the birthday-problem math there told us 64-bit
collisions are rare enough to ignore in practice (and we will add a second check
anyway).

## What to store, and why bounds are essential

The naive picture is "map position → value." But file 05 showed that a search under
a window `[alpha, beta]` usually does *not* return the exact value — it returns a
*bound*: a fail-high gives a lower bound (true value ≥ stored), a fail-low gives an
upper bound (true value ≤ stored), and only an exact node gives the true value. So a
TT entry must record not just a number but *what kind of claim* the number is.
ARCHON stores, per entry (`TtEntry` in `src/tt.rs`):

- **key16** — the low 16 bits of the Zobrist key, an in-entry tag to confirm this
  slot really belongs to the position we are asking about (the second collision
  check).
- **value16** — the searched score.
- **bound** — `Exact`, `Lower`, or `Upper` (file 05's trichotomy; the `Bound` enum).
- **depth8** — how deep the search that produced this value was. A value from a
  depth-3 search cannot substitute for a depth-10 search.
- **move16** — the best move found (for ordering; file 03's 16-bit packed `Move`).
- **eval16** — the raw static evaluation of the position, cached so we do not have
  to recompute it.
- **gen_bound8** — a byte packing the bound type, a "was this a PV node" flag, and a
  5-bit **generation** (aging counter, below).

The whole entry is **10 bytes**, and entries are grouped into 32-byte clusters
(below). Ten bytes is a deliberate squeeze — depth is one byte, key is truncated to
16 bits — because the table holds tens of millions of entries and every byte
multiplies.

Now, how a stored entry is *used* on a probe. When the search reaches a position, it
probes the TT. If it finds a matching entry whose stored depth is at least the depth
we are about to search, the stored value may let us skip the search entirely — but
only if the bound is compatible with our window:

- An **exact** entry gives the value directly.
- A **lower-bound** entry (from a fail-high) is usable only to cause a fail-high: if
  the stored lower bound is already ≥ our beta, we can cut, because the true value is
  at least that much.
- An **upper-bound** entry (from a fail-low) is usable only to cause a fail-low: if
  the stored upper bound is ≤ our alpha, we can return, because the true value is at
  most that much.

This is why the bound type is not optional metadata: reusing a fail-high value as if
it were exact would be reusing a *lower bound* as a point estimate — a lie that
corrupts the search. ARCHON's `Bound::includes_lower` / `includes_upper` helpers
encode exactly which direction each bound may be trusted in. Even when the entry is
not deep enough to cut, its `move16` still seeds move ordering, and its `eval16`
still saves an evaluation call — so a probe is almost never wasted.

There is one more correctness wrinkle: the mate-score ply adjustment from file 05. A
mate value is "distance from this node," so it is converted on the way in
(`value_to_tt`) and back on the way out (`value_from_tt`) so a mate stored at one
depth is not misread at another. And because the key exposed to the TT is
rule50-folded (file 04), two positions with identical pieces but very different
draw-clock states do not share an entry when their true values would differ near the
fifty-move boundary.

## The fixed-memory problem: replacement

Here is what makes the TT hard and interesting. It has a *fixed size* — the user
sets it (say 256 MB), and it fills up almost immediately in a real search that
visits hundreds of millions of positions. So it is not really a "remember
everything" cache; it is a "remember the most valuable things and *evict* the rest"
cache, under adversarial pressure (the search generates far more positions than fit).
Every write to a full table must decide: keep the old entry, or overwrite it? That
is the **replacement policy**, and it materially affects strength.

The two naive extremes:

- **Always replace** — overwrite whatever is there. Simple, but throws away a
  deep, expensive result to store a shallow, cheap one.
- **Depth-preferred** — only overwrite if the new entry was searched at least as
  deeply. Protects expensive results, but the table fills with stale deep entries
  from earlier in the game that will never be queried again, and never makes room
  for the current search.

Real engines blend the two, and add **aging** to solve the "stale deep entries"
problem. Each search increments a global **generation** counter; every entry
records the generation it was written in. An entry from an old generation is stale —
it belongs to a position the current search will probably never revisit — so it is
cheap to evict *regardless* of its depth. The replacement decision weighs depth
*against* staleness.

ARCHON groups entries into **clusters** of 3 (`CLUSTER_SIZE = 3` in `src/tt.rs`),
each cluster a 32-byte block that fits neatly in a cache line — so probing a
position touches exactly one cache line and reads three candidate entries. A probe
scans the three for a key match. On a write with no key match, it picks the
**replacement victim**: the entry minimizing `depth − 8 × (relative age)`. Read that
formula carefully — it says *one generation of staleness is worth 8 ply of depth*.
A slightly shallower but current entry beats a deeper but stale one; an empty slot
(depth 0) is consumed first of all. This is Stockfish's replacement scheme, and
ARCHON adopts it (the exact rules live in `TtEntry::save`). The 3-wide cluster is a
sweet spot: enough associativity that a hot position is not immediately evicted by a
colliding neighbor, but small enough to stay within one cache line.

## Indexing without a modulo

A hash table maps a key to a bucket. The textbook way is `key % num_buckets`, but a
64-bit modulo is a ~25-cycle division on the hot path, and it forces the table size
to be a power of two if you want to replace it with a mask. ARCHON avoids both with
a multiply-shift trick (`mul_hi64` in `src/tt.rs`):

```rust
fn mul_hi64(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) >> 64) as u64
}
// cluster index = mul_hi64(key, number_of_clusters)
```

The idea: treat the 64-bit key as a fraction of the way through the key space (a
number in `[0, 1)` scaled by `2^64`), and multiply by the cluster count to map it
onto `[0, num_clusters)`. Concretely, `key × num_clusters` is at most
`(2^64 − 1) × num_clusters`, so its top 64 bits are strictly less than
`num_clusters` — a valid index, uniformly distributed, for *any* table size, no
power-of-two restriction, computed with one widening multiply instead of a division.
This is a genuinely elegant piece of the design: it decouples table size from the
indexing math entirely.

## Prefetching: hiding the memory latency

The TT is tens or hundreds of megabytes — far larger than any CPU cache — so a probe
is very often a **cache miss**, a trip to main memory costing a couple hundred
cycles. If the search stalls waiting for that load every node, the TT's node savings
are partly eaten by memory latency. The fix is **prefetching**: the moment the search
knows the key of the child it is about to visit — which it computes *during*
`make_move`, before it actually recurses — it issues a `prefetch` hint telling the
CPU to start pulling that cluster's cache line into cache *now*, so that by the time
the search actually probes, the data has (hopefully) arrived. ARCHON wires this
through a `MakeMoveHooks` seam (file 04's `make_move_with`): making a move can, as a
side effect, kick off the prefetch of the resulting position's TT cluster. The
prefetch intrinsic is, notably, the *only* sanctioned `unsafe` block in the whole TT
module — everything else is safe Rust — because prefetch is a raw hardware hint with
no safe wrapper.

## Lockless sharing across threads

Modern engines search with many threads at once (file's not here, but ARCHON does
"lazy SMP": several threads search the same position, sharing one TT, and their
diversity of paths finds the answer faster). They all read and write the *same*
shared table concurrently. Locking it would serialize the hottest data structure in
the engine and destroy scaling. So the TT is **lockless**.

How can concurrent reads and writes to a shared table be safe without locks? The key
realization is that the TT is a *cache of hints*, not a source of truth. If a read
sees a torn or stale entry, the worst case is that the search does slightly redundant
work or follows a slightly worse move ordering — never a wrong *result*, because
every value pulled from the TT is re-validated by the search (a probed best move is
checked for pseudo-legality and legality before it is played; a value is only used to
cut when its bound and depth permit). So the TT can tolerate the occasional garbled
entry that a data race might produce.

ARCHON makes this precise with Rust's atomics at the weakest ordering, `Relaxed`
(`AtomicU16`, `AtomicU8`, `AtomicI16` fields in `src/tt.rs`). Relaxed atomic
loads and stores guarantee one thing: an individual field read never *tears* — you
never see half of an old `value16` and half of a new one; you get some complete
value that was written at some point. What relaxed atomics do *not* provide is
entry-wide atomicity: a probe might read a `value16` from one write and a `depth8`
from a different, concurrent write, producing an entry that never coherently
existed. The TT accepts that. The second collision check (`key16`) and the search's
own move/value re-validation catch the cases that matter, and the ones that slip
through cost only efficiency, never correctness. This "relaxed atomics plus consumer
validation" pattern is how you get a wait-free shared cache — no locks, no torn UB,
and provably-safe-enough behavior — and it is decision D14 in the project's design.
Rust's type system helps here: because the fields are atomics, the shared-reference
API (`probe`/`write` take `&self`) is sound, and the compiler will not let you
accidentally introduce a non-atomic race.

A management note that falls out of this: writes and probes go through a shared
`&self` (via the atomics), but *resizing* or *clearing* the table needs exclusive
access. ARCHON obtains that between searches on the idle UCI thread (`Arc::get_mut`),
so the table is never resized mid-search. The global generation counter is bumped
once per `go` command (one search), which is what ages the previous search's entries.

## The TT in the search loop

Pulling it together, here is a node's interaction with the TT:

```
on entering a node:
    entry = TT.probe(key)
    if entry matches and entry.depth ≥ our depth:
        if entry.bound permits a cutoff against [alpha, beta]:
            return entry.value        # skip the search entirely
    tt_move = entry.move16            # even if we cannot cut, use it for ordering
    ... search the position (PVS, file 05), trying tt_move first ...
    ... during make_move of each child: prefetch that child's TT cluster ...
on leaving the node:
    TT.save(key, best_value, bound, depth, best_move, static_eval, generation)
```

The probe can end the node instantly (a cutoff), or merely improve its ordering. The
save records what this search learned, subject to the replacement policy, tagged with
the current generation so the next search can age it out. And this same machinery,
threaded through iterative deepening (file 06), is what carries each iteration's
best moves down to the next — the ordering feedback loop that makes deepening pay.

## What you now know

You know that transpositions make the search space a DAG, that re-searching
converged positions is a large fraction of the naive work (~47% in ARCHON's
measurement), and that the transposition table caches search results — keyed by the
Zobrist hash — to eliminate it, while doubling as a move-ordering oracle. You
understand *why* an entry must store its bound type (`Exact`/`Lower`/`Upper`) along
with value, depth, and best move, and how each bound may be trusted only in one
direction. You know the fixed-memory replacement problem and ARCHON's answer:
3-wide cache-line clusters, a depth-versus-generation replacement score where one
age is worth eight ply, and generation-based aging of stale entries. You have seen
the `mul_hi64` multiply-shift that indexes any table size without a division, the
prefetch that hides the cache-miss latency of a huge table, and the lockless design
— relaxed atomics that cannot tear plus consumer re-validation — that lets many
threads share the table wait-free without ever producing a wrong result.

**Next:** `08-classical-evaluation.md` — search propagates leaf values, and we have
so far treated `evaluate` as a black box. Now we open it, in its classical form: how
material, piece placement, and game phase combine into a single number by hand-tuned
tables — and why, despite decades of refinement, that approach hit a ceiling that
pushed the whole field toward learned evaluation.
