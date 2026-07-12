# NNUE: input features, threats, and network shape (Reckless v0.10.0-dev)

Source (read in full): `src/nnue.rs`; `src/nnue/accumulator.rs`;
`src/nnue/accumulator/psq.rs`; `src/nnue/accumulator/threats.rs`;
`src/nnue/accumulator/threats/threat_index.rs`;
`src/nnue/accumulator/threats/scalar.rs`;
`src/nnue/accumulator/threats/vectorized.rs` (+ `vectorized/avx2.rs`,
`vectorized/avx512.rs`); `src/nnue/forward/scalar.rs`,
`src/nnue/forward/vectorized.rs`; `src/nnue/simd/{avx2,avx512,scalar}.rs`;
`build/build.rs`; `src/board.rs` (`BoardObserver`), `src/board/makemove.rs`
(hook call sites), `src/numa.rs` (`NumaReplicable`). Reckless is AGPLv3
(D27 mirror) — everything below is prose/own-words pseudocode; no Reckless
code is reproduced, only isolated constants/formulas where exactness matters
and short (< 5 line) illustrative snippets I wrote myself.

Our comparison points: `design/11-evaluation.md` §3.4 (our NNUE plan),
`design/15-nnue-datagen.md` §3.2 (the `Chess768`/`bulletformat` contract),
`design/16-research-data-platform.md`. Stockfish's own NNUE internals are
**not** covered by our existing `stockfish_reference/*.md` notes — those
predate NNUE study (confirmed: `stockfish_reference/NOTES.md` only lists
`nnue/` in the file-map table, no dedicated note) — so this is the first
NNUE-architecture note in the project. Where I compare against Stockfish,
I'm relying on what `design/11` already records about SF for calibration
(§3.4.1: "current SF... threat-augmented feature set, L1 = 1024, deep
15/32-neuron tail with sparse propagation, dual big/small nets"), not on a
fresh read of SF's `nnue/` source.

## 1. Big picture

Reckless runs **two parallel feature transformers that write into one
shared 768-wide accumulator per perspective**, not one feature set:

```
 board  ──────────────┬────────────────────────────┐
                       │                            │
             piece-square (PST) features   pairwise "threat" features
             (king-bucketed HalfKA-ish)    (attacker/victim relational)
                       │                            │
                 W_psq  (i16, 10×768 rows)    W_thr  (i8, 66864 rows)
                       │                            │
                       └──────────► + ◄──────────────┘
                         per-perspective accumulator
                              (i16 × 768, × 2 povs)
                                     │
                    clamp, split in half, cross-multiply ("pairwise mul")
                                     │
                          768-wide u8 vector (both povs concatenated: 384+384)
                                     │
                     sparse int8 L1 (only nonzero 4-lanes walked)  → 16 floats
                                     │
                            L2 (16→32, float, ReLU[0,1])
                                     │
                            L3 (32→1, float) × NETWORK_SCALE (380)
```

The PST half is a fairly standard king-bucketed, horizontally-mirrored
piece-square feature set (HalfKA family). The threat half is the
distinctive part: instead of "piece P is on square s", a threat feature
encodes "piece P on square s attacks (or defends) piece Q on square t" —
a *relational* feature over pairs of pieces, not a per-square occupancy
feature. Both feature sets are quantized and summed into the **same**
accumulator width (`L1_SIZE = 768`, `src/nnue.rs:87`), so architecturally
this is equivalent to one wide feature transformer over the union of both
feature spaces — additive fusion of two linear maps is linear, so splitting
it into two update paths is purely an implementation/perf choice (different
sparsity, different natural update cadence — see §5), not a change in what
the network can represent versus a single concatenated-input FT.

## 2. Network topology & dimensions

| Constant | Value | Meaning | Cite |
|---|---|---|---|
| `L1_SIZE` | 768 | accumulator width per perspective (before pairwise-mul); also, confusingly, the width of the *post-activation* FT output (384 own + 384 other, after halving-then-doubling — see §6) | `nnue.rs:87` |
| `L2_SIZE` | 16 | first float layer width | `nnue.rs:88` |
| `L3_SIZE` | 32 | second float layer width | `nnue.rs:89` |
| `INPUT_BUCKETS` | 10 | king buckets for the PST feature set | `nnue.rs:84` |
| `OUTPUT_BUCKETS` | 8 | output head selected by total piece count | `nnue.rs:85` |
| `FT_QUANT` | 255 | FT fixed-point scale | `nnue.rs:91` |
| `L1_QUANT` | 64 | L1 weight fixed-point scale | `nnue.rs:92` |
| `FT_SHIFT` | 9 | right-shift in the FT pairwise-mul (differs by ISA: `u32` on AVX-512, `i32` elsewhere) | `nnue.rs:94-97` |
| `NETWORK_SCALE` | 380 | final centipawn-ish scale multiplier | `nnue.rs:82` |
| `DEQUANT_MULTIPLIER` | `2^FT_SHIFT / (FT_QUANT² · L1_QUANT)` ≈ 0.0001230 | converts the int32 L1 pre-activation back to float before the L2/L3 float tail | `nnue.rs:99` |

Shape summary: `(768 PST + 66864 threat) → 768 (shared, fused) → [pairwise-mul] → 768 → 16 → 32 → 1`,
8 king buckets... no — 10 input (king) buckets, 8 output (material) buckets.
This is a genuine 4-layer net past the feature transformer (L1→L2→L3→out),
matching what `design/11` already flags as the eventual destination ("deep
15/32-neuron tail" — Reckless's actual tail is 16/32, close enough to be the
same design lineage).

### 2.1 Parameter-count accounting (own calculation, not read off a shipped file — no `.nnue` file is present in this clone; `build/build.rs:14-15` names it `v60-7f587dfb.nnue`, fetched from a GitHub Releases URL at build time)

Computed directly from the `Parameters` struct field types (`nnue.rs:356-367`):

| Table | Shape | Element | Bytes |
|---|---|---|---|
| `ft_threat_weights` | `66864 × 768` | i8 | 51,351,552 |
| `ft_piece_weights` | `7680 × 768` (= 10 buckets × 768) | i16 | 11,796,480 |
| `ft_biases` | `768` | i16 | 1,536 |
| `l1_weights` | `8 × 16 × 768` | i8 | 98,304 |
| `l1_biases`, `l2_weights`, `l2_biases`, `l3_weights`, `l3_biases` | small | f32 | ~19,000 combined |
| **Total** | | | **≈ 63.27 MB ≈ 60.3 MiB** |

**The threat-feature weight table is ~81% of the entire network's size**,
and it's already the *smaller*-precision table (i8, not i16) — at i16 parity
with the piece table it would be ~102 MB alone. This is the single most
important sizing fact for anyone considering porting threat features: the
input-feature explosion dominates the model, not the L2/L3 tail.

## 3. Feature set 1 — king-bucketed piece-square (PST) inputs

`accumulator/psq.rs`. This is architecturally almost identical to our own
`design/11` §3.4.1 v1 formula (`idx = ((c != p) as usize*6+pt as usize)*64 + rel_sq(p,sq)`)
— it's the same HalfKA-style shape, just with king buckets and a horizontal
mirror layered on:

```
# my own restatement of pst_index() (psq.rs:204-211)
flip      = (7 if own_king.is_kingside() else 0) XOR (56 if pov == Black else 0)
bucket    = INPUT_BUCKETS_LAYOUT[own_king_square XOR flip]     # 0..=9
color_bit = 384 if piece.color() != pov else 0
feature   = bucket*768 + color_bit + piece_type*64 + (square XOR flip)
```

`flip` folds two independent mirrors into one XOR: file-mirror when the own
king sits on the e/f/g/h files (`is_kingside`), and rank-mirror when `pov`
is Black (so every position is always evaluated as if "my king, on my
turn" were White). `INPUT_BUCKETS_LAYOUT` (`nnue.rs:101-111`) is a 64-entry
table, one bucket id per king square, and it is markedly **non-uniform**:

```
rank1: 0 1 2 3 3 2 1 0      (4 distinct buckets, mirrored pairs — files ab/cd)
rank2: 4 5 6 7 7 6 5 4      (4 more distinct buckets)
rank3: 8 8 8 8 8 8 8 8      (one bucket for the entire rank)
rank4-8: 9 9 9 ... 9        (one bucket for the other 40 squares)
```

So 8 of the 10 buckets are spent entirely on **ranks 1–2** (where a castled
or home-row king actually lives most of the game), and ranks 3–8 combined
get only 2 buckets total. This is a strong, explicit bet that king-safety
structure only needs fine resolution near the back ranks — worth noting as
a deliberately unbalanced allocation, unlike a uniform grid.

**Refresh caching (Finny table).** `AccumulatorCache` (`accumulator.rs:13-41`)
holds one cached accumulator row per `(pov, is_kingside, bucket)` — i.e.
`2 × 2 × 10 = 40` cached entries — each remembering the `pieces`/`colors`
bitboards it was last built from. `PstAccumulator::refresh` (`psq.rs:33-67`)
diffs the *current* board against the cache entry's remembered occupancy to
get a minimal add/sub feature list, applies just that delta to the cached
row, and copies the result out — turning "recompute from scratch" into "diff
against the nearest same-bucket position ever seen," which is exactly the
technique our `design/11` §3.4.1 upgrade-path note anticipates ("an
accumulator-cache table becomes worthwhile" once king buckets exist). Only
king moves that change bucket **or** kingside/queenside half force this path
(`can_update_pst`, `nnue.rs:242-262`); everything else is a plain forward
delta walk (`update`, `psq.rs:105-138`), 1 add/1 sub for quiet moves, up to
2/2 for capture/castle/en-passant/promotion — the same shape as our own
planned `DirtyPiece` (`design/11` §3.4.4: `adds`/`subs`, ≤2 each).

An AVX-512VBMI2 variant of `push_features` (`psq.rs:80-103`) replaces the
per-set-bit loop with a `_mm512_maskz_compress_epi8` bit-compress against an
iota vector — extracting the sparse list of set-square indices from a
bitboard without a software popcount/bitscan loop. Neat but ISA-gated; not
something we need before AVX-512 is in scope for us.

## 4. Feature set 2 — threats (the distinctive part)

### 4.1 What a threat feature encodes

A threat feature fires for every `(attacker, attacked)` pair currently
present on the board where the attacker's piece-attack pattern from its
square reaches the attacked piece's square — i.e., every attack **and**
every defense relationship simultaneously (an attacker defending its own
piece is the same kind of edge as attacking an enemy piece; only the
relative-color bit differs). This is a much richer signal than occupancy:
the network is handed "rook R1 attacks knight N1" directly as a feature
rather than having to infer it from raw piece-square activations, the way a
plain HalfKA net must.

### 4.2 Index derivation — my own restatement of `threat_index()` (`threat_index.rs:103-117`)

```
fn threat_feature_id(attacker, attacker_sq, victim, victim_sq, pov, mirror) -> Option<u32>:
    a_sq = relative_to(attacker_sq, pov) XOR (7 if mirror else 0)   # canonical squares
    v_sq = relative_to(victim_sq,  pov) XOR (7 if mirror else 0)

    # relabel "which side owns this piece" as mine(0)/theirs(1) by XOR-ing the
    # color bit with pov — works for free because Piece's own encoding
    # interleaves color as the low bit (WhitePawn=0, BlackPawn=1, ...), so
    # XOR-by-pov is exactly a friend/enemy relabeling that leaves piece TYPE
    # (the upper bits) untouched.
    a = attacker XOR pov
    b = victim   XOR pov

    pair = PAIR_TABLE[a][b]                 # built once at startup, §4.3
    if pair.excluded: return None
    if pair.same_type_symmetric and a_sq < v_sq: return None   # dedup mutual pairs

    return pair.block_base
         + EDGE_PREFIX[a][a_sq]             # # of a's possible edges before a_sq
         + EDGE_RANK[a][a_sq][v_sq]         # dense rank of v_sq among a_sq's own targets
```

Three compression tricks stack here:

1. **Target-category merging.** Not every `(attacker type, victim type)` pair
   gets its own bucket — a `PIECE_INTERACTION_MAP[6][6]` (own paraphrase of
   `threat_index.rs:35-42`) maps `(attacker_type, victim_type)` to a small
   category index (0..4) or `-1` (excluded entirely). Decoded categories per
   attacker (which victim types get **no** feature at all):

   | Attacker | Victim types with **no** feature |
   |---|---|
   | Pawn | Bishop, Queen, King |
   | Knight | King |
   | Bishop | Queen, King |
   | Rook | Queen, King |
   | Queen | King |
   | King | Queen, King |

   Attacking the enemy king is excluded for *every* attacker type — sensible,
   since "giving check" is a transient, always-forced-response state, not a
   generalizable static "threat" signal. The Pawn/Bishop/Rook exclusions
   against Queen (and Pawn against Bishop) are less obviously motivated from
   the code alone — I could not find documentation of the exact rationale;
   plausibly a deliberate parameter-count trim (own-attacks-queen already
   correlates with SEE/material elsewhere, or was found low-value in
   ablation). Flagging as an open question rather than asserting the reason.

2. **Same-type mutual-pair dedup.** When attacker and victim are the same
   piece type (and it's either an enemy pair, or non-pawns), the relationship
   is symmetric — "A attacks/defends B" and "B attacks/defends A" describe
   the same physical fact — so only the direction with the numerically larger
   square index is kept, halving that slice of the table. (Encoded as a
   carry trick in the original: a spare bit gets bumped into the "excluded"
   bit exactly when the pair is symmetric *and* attacker-square < victim-square;
   I'm describing the *effect*, not the bit-packing.)

3. **Prefix-sum + dense-rank compression** (`EDGE_PREFIX`/`EDGE_RANK` above,
   `PIECE_OFFSET_LOOKUP`/`ATTACK_INDEX_LOOKUP` in the source,
   `threat_index.rs:30-31,92-100`). For a given attacking piece type and
   from-square, the set of reachable squares on an empty board is fixed;
   `EDGE_RANK[attacker][from][to]` is simply "how many of `attacker`'s
   possible targets from `from` have a lower square index than `to`" — a
   popcount-below-a-mask trick, the same idea as kindergarten-bitboard /
   kingside compression tricks used elsewhere in engines for turning a
   sparse attack pattern into a dense small integer without a search or a
   hash.

### 4.3 Feature-space size — validated by independent recomputation

The total feature-index space is **66864** (`nnue.rs:358`,
`[[i8; L1_SIZE]; 66864]`). I recomputed this independently from first
principles (empty-board attack-edge counts per piece type, summed over all
64 squares — pawns restricted to ranks 2–7 per `threat_index.rs:60`) to
confirm the derivation and got an exact match:

| Attacker type | Edges (empty board, both colors' squares) | Target categories × 2 colors | Contribution |
|---|---|---|---|
| Pawn | 84 | 6 | 1,008 |
| Knight | 336 | 10 | 6,720 |
| Bishop | 560 | 8 | 8,960 |
| Rook | 896 | 8 | 14,336 |
| Queen | 1,456 | 10 | 29,120 |
| King | 420 | 8 | 6,720 |
| **Total** | | | **66,864** |

(Both colors of attacker included — the table above already reflects the
`× 2` from `for piece_color in [White, Black]` at
`threat_index.rs:50`.) The Queen alone is 43.5% of the whole feature space,
Rook another 21.4% — sliders with long reach dominate the index space simply
because they have the most possible `(from, to)` edges to enumerate, not
because of any special weighting.

### 4.4 Incremental maintenance — scalar path (`accumulator/threats/scalar.rs`)

Threat deltas are generated at the *board-mutation* level via a
`BoardObserver` trait (`board.rs:597-609`) with three hooks —
`on_piece_move`, `on_piece_mutate` (promotion, in place), `on_piece_change`
(add/remove, e.g. castling rook, en-passant victim) — called directly from
`make_move`'s generic `<T: BoardObserver>` parameter
(`board/makemove.rs:32` and call sites through line ~91). This is the same
architectural shape as our own `design/04` §3.10 `MakeMoveHooks` /
`make_move_with` seam — a monomorphized, zero-cost-when-unused hook attached
to the move-application code path, not bolted on after the fact.

For each square touched by a move (the vacated origin and the newly
occupied destination), the scalar generator (my own restatement of
`push_threats_single`, `scalar.rs:18-55`) records:

```
fn record_square_touch(deltas, board, occ, piece, sq, arriving):
    # (a) direct: what this piece now/no-longer attacks from sq
    for target in attacks(piece, sq, occ) & occ:
        deltas.push(edge(piece@sq -> board.piece_at(target)@target), arriving)

    # (b) x-ray ripple: sliders whose ray passes through sq see one square
    #     further (departure) or are newly blocked (arrival) — plus the
    #     slider now also has a relationship with the piece sitting on sq
    #     itself
    for slider in (bishop_attacks(sq,occ) | rook_attacks(sq,occ))
                  & (board's diagonal/orthogonal sliders) & occ:
        if let Some(far) = first_square_beyond(slider, sq, occ):
            deltas.push(edge(slider -> far), NOT arriving)   # flips old vs new
        deltas.push(edge(slider -> piece@sq), arriving)

    # (c) reverse: short-range attackers of sq (pawns/knights/king) now
    #     (dis)attack the piece sitting on sq
    for attacker in (pawn_attackers(sq) | knight_attackers(sq) | king_attackers(sq)) & occ:
        deltas.push(edge(attacker -> piece@sq), arriving)
```

A single moved piece can therefore ripple into many unrelated relationship
changes (every slider that was x-rayed through the vacated/occupied square,
every short-range piece adjacent to either square) — the delta buffer is
sized for up to **80** raw events per move
(`ArrayVec<ThreatDelta, 80>`, `threats.rs:61`), versus our own planned
`DirtyPiece`'s ≤2 add/≤2 sub. `push_threats_on_mutate` (in-place piece-type
change: promotion) is similar but doesn't move anything — it just
subtracts-then-adds every relationship touching the fixed square for the old
vs. new piece identity (`scalar.rs:57-89`). Each raw delta is stored as an
absolute (piece, from, attacked, to, add/sub) tuple in a packed `u32`
(`ThreatDelta`, `threats.rs:21-56`) — perspective-agnostic; the
perspective/mirror-dependent `threat_index()` lookup (§4.2) happens later,
once per (accumulator, pov) at consumption time, not at generation time —
so the same raw delta list serves both White's and Black's accumulator
update.

I could not fully pin down, with certainty, why `push_threats_on_move`
(`scalar.rs:12-16`) computes one shared "occupancy with the destination
square's bit toggled" and reuses it for **both** the origin- and
destination-square passes, rather than two different occupancy snapshots —
plausibly a one-pass reuse for the common case where the two squares aren't
mutually visible along any ray, but I'm flagging this as read-but-not-fully-
verified rather than asserting the exact reasoning.

### 4.5 Incremental maintenance — vectorized path (`threats/vectorized.rs`, `vectorized/avx2.rs`, `vectorized/avx512.rs`)

Same relationships, computed branch-free with SIMD instead of bitboard
lookups + a manual x-ray loop:

1. A precomputed **per-origin-square byte-permutation table**
   (`RAY_PERMUTATIONS`, `vectorized.rs:17-45`) gathers the 64-byte mailbox
   into 8 groups of 8 bytes, one group per ray direction (N/NE/E/SE/S/SW/W/NW),
   ordered outward from the origin, built via 0x88-style off-board detection
   (an invalid hop marks that lane invalid instead of wrapping around the
   board edge).
2. The mailbox (one byte per square, a `Piece` tag) is loaded as SIMD vectors
   and shuffled through that permutation (`board_to_rays`,
   `vectorized/avx2.rs:66-90`, using `pshufb`-style byte shuffles + a
   128-lane blend to gather across both halves of the 64-square board).
3. Each mailbox byte is translated through a 16-entry lookup table into a
   small per-piece-class bitmask (`PIECE_TO_BIT_TABLE`, `vectorized.rs:62-71`).
4. `closest_on_rays` (`vectorized/avx2.rs:28-38`) applies the classic SWAR
   "isolate the lowest set byte per 8-byte lane" trick (`o ^ (o - mask)`,
   the same family of bit-trick sliding engines use for single-ray attack
   generation) — but generalized to all 8 rays from one square at once,
   yielding the nearest occupied square along every ray simultaneously
   instead of walking each ray with a loop.
5. Small precomputed per-ray-direction masks (`RAY_ATTACKERS_MASK`,
   `RAY_SLIDERS_MASK`, `vectorized.rs:73-107`) then classify, per ray,
   whether the nearest piece attacks the origin, is attacked by it, or is a
   slider whose "far side" also needs an x-ray update — all as masked SIMD
   compares, no branches.

This is a genuinely clever vectorized reformulation of "what changed near
this square" — worth recognizing as an engineering achievement distinct from
the feature-set idea itself, but it's deep, ISA-specific machinery (three
near-full-page files of permutation tables and shuffle sequences) that we
would only need if we adopt threat features **and** want AVX2/AVX-512
parity; a scalar-only threat implementation (§4.4's shape) is a much smaller
lift and is where any first attempt on our side should start.

### 4.6 Refresh vs. update triggers — an asymmetry worth noting

`can_update_pst` (`nnue.rs:242-262`) forces a full PST refresh whenever the
own king crosses a *bucket* boundary (any of the 10 `INPUT_BUCKETS_LAYOUT`
regions) **or** the kingside/queenside mirror half. `can_update_threats`
(`nnue.rs:264-284`) forces a threat refresh only on the mirror-half change
— threats have no king-bucket concept at all, only the horizontal mirror —
so **the threat accumulator needs a full refresh strictly less often than
the (much cheaper-to-store) PST accumulator does**, despite threats owning
by far the larger weight table. Both checks reuse the *same* underlying
per-ply delta record (the `PstDelta`'s `mv`/`piece` fields, not a
threat-specific one) to answer "did the king change halves" — a small but
neat piece of code reuse: one source of truth for "was this a
mirror-changing king move," shared by both accumulator kinds
(`nnue.rs:270` reads `self.pst_stack[i].delta` inside `can_update_threats`).
Also notable: unlike the PST side, the threat `refresh()` (`threats.rs:74-139`)
has **no Finny-style cache** — it's a plain loop over every occupied square
computing `attacks(...) & occupancies`. This is presumably fine because,
per the above, this path triggers only on mirror flips, which are rarer
than bucket changes.

## 5. Fusion + activation — the "pairwise multiply" trick

`activate_ft` (`forward/scalar.rs:9-25`, my own restatement):

```
fn activate_ft(pst_acc, threat_acc, side_to_move) -> [u8; 768]:
    out = [0u8; 768]
    for half in [own_perspective, other_perspective]:      # side-to-move first
        combined = clamp(pst_acc[half] + threat_acc[half], 0, FT_QUANT)   # 768 values
        left, right = combined[0..384], combined[384..768]
        for i in 0..384:
            out[half_offset + i] = ((left[i] as i32 * right[i] as i32) >> FT_SHIFT) as u8
    return out
```

Two things to notice. First, PST and threat contributions are **summed
elementwise before clamping** — the two feature sets share one linear
target space, confirming §1's "additive fusion = one wide FT over the
union of feature spaces." Second, instead of a plain squared-clipped-ReLU
(`clamp(x,0,QA)²`, which is what our `design/11` §3.4.3 currently specifies),
Reckless clamps the accumulator's two halves independently and **multiplies
corresponding elements across the halves** — a bilinear/gated unit rather
than a self-product. This costs the same one multiply-per-output-unit as
SCReLU, halves the per-perspective width in the process (768 → 384 per
perspective, recovered by concatenating both perspectives back to 768
total), and is plausibly more expressive than squaring a single value
against itself, since each output is now a product of two *independently
learned* projections. This is a genuine "network-shape" idea distinct from
the threat feature set — it's an activation-function choice, portable
independently of whether we ever adopt threats at all.

## 6. Rest of the forward pass

`forward/scalar.rs:27-128` (also mirrored, SIMD, in `forward/vectorized.rs`,
required by the file's own comment to be bit-exact with the scalar path in
order of float operations — FMA vs. separate mul+add matters, per
`forward/scalar.rs:58-59`):

- **Sparse L1** (`propagate_l1`): the FT output is packed 4-bytes-at-a-time
  into `i32`s (192 chunks over 768 bytes); `find_nnz` scans for chunks that
  are entirely zero and skips them in the dot product. The scalar version
  just checks each chunk directly; the vectorized version additionally uses
  a 256-entry `SparseEntry` lookup table (`nnue.rs:125-166`, built once at
  `Network::new`) mapping a byte's bit pattern to the list of set-bit
  positions, for fast per-lane index extraction — the standard "sparse
  input propagation" trick also used elsewhere in the NNUE literature (this
  part is *not* Reckless-distinctive; it's the same idea `design/11` §3.4.6
  earmarks for when our own L1 widens past a size where a dense dot product
  is cheap).
- **Dequantization bridge**: L1's int32 accumulation is converted to float
  via `(pre_activation as f32).mul_add(DEQUANT_MULTIPLIER, bias)`, then
  clamped to `[0, 1]` — a clipped-ReLU in float space, not int8 SCReLU.
- **L2 (16→32) and L3 (32→1)** run entirely in `f32`, plain
  `mul_add`-accumulated affine layers with `[0,1]` clamps between them
  (`propagate_l2`/`propagate_l3`, `forward/scalar.rs:67-108`). This
  int8-sparse-L1-then-float-tail hybrid is exactly the concrete recipe our
  own `design/11` §3.4.1 gestures at ("deep 15/32-neuron tail with sparse
  propagation") without yet specifying a quantization bridge for it — this
  note supplies that missing concrete numeric recipe.
- **Output bucket selection**: `OUTPUT_BUCKETS_LAYOUT[popcount(all occupancy)]`
  (`nnue.rs:113-123`), a hand-shaped, **non-uniform** 33-entry table: one
  bucket for 0–8 pieces (endgames), then narrowing bands of 3–4 pieces per
  bucket up through 32 — unlike a uniform `(popcount-2)/4` formula (what our
  `design/11` §3.4.1 currently plans). Finer resolution sits in the
  middlegame piece-count range; the sparsest-material band is deliberately
  coarse.
- **Final scale**: `(l3_out * NETWORK_SCALE) as i32`, `NETWORK_SCALE = 380`.

## 7. Net file format & loading

No versioned container at all: `Parameters::embedded()`
(`nnue.rs:369-373`) does `std::mem::transmute(*include_bytes!(env!("MODEL")))`
directly onto the `#[repr(C)]` `Parameters` struct — the shipped `.nnue`
file's bytes must match Rust's field layout (including the `#[repr(align(64))]`
padding `Aligned<T>` inserts between big tables) **exactly**, with no magic
number, version tag, dimension check, or checksum. `build/build.rs:14-15,58-66`
downloads a named release asset
(`v60-7f587dfb.nnue`, from a `RecklessNetworks` GitHub Releases URL) at
build time unless `EVALFILE` is set, and embeds whatever bytes result. This
is fast (zero parse cost, just a pointer cast) and simple, but brittle:
any struct layout drift between the trainer's exporter and this crate's
`Parameters` definition is undetectable at load time (there's no local
`netpack`/exporter tool in this repo — it must live in a companion training
repo, kept in lockstep by convention only). **Our own `design/11` §3.4.5
CENN format (magic + version + dims + activation id + checksum) is already
strictly safer here** — nothing to port, just confirmation we made the
right call.

## 8. NUMA replication (forward-pointer, not P13)

`ParametersHandle`/`ParametersStorage` (`nnue.rs:387-427`) wraps the
embedded net in either a `&'static` reference (single-node) or a per-node
`Arc<Parameters>` via a `NumaReplicable` trait (`numa.rs:16-22`,
`allocate`/`allocate_shared`) — each NUMA node gets its own physical copy of
the (large, read-mostly) weight tables so lazy-SMP worker threads never
cross-node-fault on every accumulator update. Not relevant to P13 itself,
but directly relevant to `design/13-threading-smp.md` (P11) once we have
weights large enough (a threat table, in particular, at ~50 MB, is exactly
the kind of table that would benefit from this) to make cross-node access
costly.

## 9. Distinctive vs. Stockfish / vs. us — summary

| Aspect | Reckless | Stockfish (per `design/11`'s existing SF notes) | Us (`design/11` v1 plan) |
|---|---|---|---|
| Base feature set | King-bucketed HalfKA-ish PST (10 buckets, mirror) | HalfKAv2_hm (king-bucketed + mirror), longer-established scheme | Unbucketed 768 HalfKA-shape (bucket/mirror are named upgrade-path items) |
| Threat features | Yes — ~51 MB i8 table, 66864-wide, own compression scheme | `design/11` names "threat-augmented" as SF's current generation too, but we haven't read SF's own scheme | Not present; named as the eventual "destination, not the first step" |
| Activation | Split-accumulator pairwise multiply (bilinear) | (not verified from our SF notes) | SCReLU (`clamp(x,0,QA)²`) planned |
| Hidden width | 768 shared (fused PST+threat) | 1024 (per `design/11`) | 256, upgrade path to 512/1024 |
| Deeper tail | Yes: 16→32→1, int8-sparse-L1 then float L2/L3 | "deep 15/32-neuron tail with sparse propagation" (per `design/11`) | Not yet designed; single 512→1 output layer in v1 |
| Output buckets | 8, non-uniform piece-count bands | (unknown from our notes) | 8, uniform `(popcount-2)/4` planned |
| Net file format | Raw struct transmute, no header/checksum | (SF has its own versioned format, not examined here) | Own CENN format: magic+version+checksum — already safer than Reckless |
| Incremental update model | Ply-indexed stack, walk-back to nearest accurate ancestor, explicit per-ply invalidation | Accumulator lives in `StateInfo` (freshness is structural) | Ply-indexed stack + explicit invalidation via `on_dirty` hook — same shape as Reckless, convergent design |
| Hook mechanism | Generic `BoardObserver` trait threaded through `make_move<T>` | N/A (different codebase shape) | Generic `MakeMoveHooks` trait threaded through `make_move_with` — same shape, independently arrived at |

The incremental-update architecture (ply-indexed accumulator stack +
explicit per-ply "this entry's delta hasn't been applied yet" flag + walk
back to nearest valid ancestor) is **essentially the same idea already
locked into our `design/11` §3.4.4 / `design/04` §3.10 hook seam** —
convergent design, not something to import. The two things genuinely absent
from our plan are (a) the threat feature set itself, and (b) the deeper
int8→float hybrid tail with a concrete dequantization bridge.

## 10. Adapt to us

Ordered roughly by effort, cheapest first. Each references the owning doc
section and roadmap phase (all are P13, "NNUE", per `design/00-master-plan.md`,
except where noted).

1. **Horizontal king-mirror, no buckets yet.** Always transform squares by
   `XOR 7` when the own king sits on files e–h (§3's `flip` component, minus
   the bucket lookup). Free-ish: halves the effective input redundancy the
   FT has to learn (mirrored positions collapse to one weight row) with zero
   new data-structures — no cache needed since there's still only one
   "bucket." **Doc**: `design/11` §3.4.1 (already named as the first
   upgrade-path step). **Effort**: low. **Impact**: likely small-but-free
   Elo/data-efficiency; nearly a pure win, easy first SPRT-gated increment
   once the plain 768 net (D25) is validated.

2. **King input buckets + Finny-table refresh cache.** Reckless's concrete,
   validated `INPUT_BUCKETS_LAYOUT` shape (§3: 8 buckets crammed into ranks
   1–2, 2 buckets for the rest of the board) is a good starting point to
   imitate the *shape* of (not copy verbatim) rather than inventing bucket
   boundaries from scratch. **Doc**: `design/11` §3.4.1 upgrade path
   ("HalfKAv2_hm-style... accumulator-cache table becomes worthwhile").
   **Effort**: medium (needs `AccumulatorCache`-equivalent: one cached row
   per `(pov, mirror-half, bucket)`, diffed against remembered occupancy on
   refresh — §3 gives the exact mechanics to reimplement independently).
   **Impact**: the step that unlocks meaningfully wider/deeper nets down the
   line; moderate-to-large Elo based on how consistently other engines gain
   from it.

3. **Split-accumulator pairwise-multiply activation**, in place of the
   currently-planned SCReLU. Pure activation-function swap once the FT/L1
   pipeline exists: clamp each half of the (now-shared) accumulator, then
   elementwise-multiply left half by right half instead of squaring one
   value. Same compute shape (one multiply per output unit), potentially
   more expressive (bilinear vs. self-product). Worth checking whether our
   external trainer (`bullet`, `design/11` §3.4.7 / `design/15`) already has
   a name for this activation (a pairwise/"paired-CReLU-mul" option is
   plausible given how widely the trick seems to travel) before designing
   our own. **Doc**: `design/11` §3.4.3 (quantization/activation choice).
   **Effort**: low-medium (mostly a training-recipe + forward-pass change,
   no new data structures). **Impact**: unclear magnitude, cheap experiment,
   plausible free Elo since a top-2 engine ships it.

4. **Int8-sparse-L1 → float-L2/L3 hybrid tail**, with the exact
   dequantization-bridge formula this note supplies
   (`DEQUANT_MULTIPLIER = 2^shift / (QA² · QB)`, clamp `[0,1]` between float
   layers). This is the concrete recipe for the "deep tail" our own
   `design/11` §3.4.1 already names as the destination but doesn't yet spec
   numerically. **Doc**: `design/11` §3.4.1 / §3.4.6 (upgrade path, post-v1).
   **Effort**: medium (new layer sizes, a training-side quantization-range
   decision, a scalar/SIMD bit-exactness test in the same spirit as our
   existing SCReLU plan). **Impact**: needed once we go past a single output
   layer at all; this and #2 are the two "grow the net" steps, both
   SPRT-gated per D25.

5. **Non-uniform output-bucket boundaries.** Trivial to A/B once output
   buckets exist at all — swap the `(popcount-2)/4` uniform table for a
   hand-shaped one (or even just copy the *shape* — narrow bands mid-game,
   wide bands at the endgame/opening extremes — and retune boundaries
   against our own data). **Doc**: `design/11` §3.4.1. **Effort**: trivial.
   **Impact**: small, cheap experiment, no architecture change.

6. **Threat input features — the big one.** Concretely feasible, but the
   most expensive item on this list, and the one with real infrastructure
   fallout, not just a network change:
   - **Feature-index design**: don't copy Reckless's exact bit-packed
     `PiecePair`/prefix-sum scheme verbatim (AGPL + it's fairly intricate);
     the *shape* to reimplement independently is: (a) merge some
     attacker/victim type combinations into shared "category" buckets to
     trim the space, (b) drop king-as-victim entirely (transient, not a
     generalizable static feature), (c) dedup mutual same-type pairs by
     keeping one square-ordering direction, (d) use a popcount-below-mask
     dense-rank trick to compress `(from, to)` into a small dense index
     instead of a full 64×64 table per piece type. §4.2–4.3 give the exact
     arithmetic to replicate the sizing math (66864 = Σ edges×categories×2)
     with our own category choices.
   - **Weight-table size**: budget for a large table regardless of exact
     scheme — Reckless's is ~51 MB in i8 alone (§2.1), ~81% of the whole
     net. This has direct consequences for net-file size, embed-at-build-time
     cost, and cache behavior during search.
   - **Incremental maintenance**: our `MakeMoveHooks`/`DirtyPiece` seam
     (`design/11` §3.4.4, `design/04` §3.10) would need a materially richer
     payload than ≤2 add/≤2 sub — Reckless budgets up to 80 raw relationship
     events per move (§4.4). We already have the attack-table primitives
     (`attacks(pt, sq, occ)`, `attackers_to_exist` per `CLAUDE.md`'s
     `movegen` section) to compute the same "who now attacks/defends whom
     near this square" ripple; a scalar-only version (no vectorized ray-SWAR
     machinery, §4.5, which is a large, ISA-specific undertaking on its own)
     is the right-sized first attempt.
   - **Datagen/training pipeline**: our `design/15` §3.2.3 pins
     `bulletformat::ChessBoard`/`Chess768` specifically *because* it's a flat
     768-input record — a threat-augmented feature set is not representable
     in that record type at all. Adopting threats means either (a) a custom
     sparse-feature loader (`bullet` supports more general input types per
     its docs; would need investigation beyond what this note covers) or
     (b) computing threat-feature activations at train time from the board
     and feeding bullet a wider generic sparse-input format. Either way,
     `design/15` §3.2 would need a real amendment, not just a network-shape
     change — flag this cost explicitly when scoping the work, and note
     `design/16`'s layered Observation-Store model would still hold (a new
     Derived-layer projection for threat-feature corpora), so the platform
     doesn't need to change, just grow a new projection.
   - **Doc**: `design/11` §3.4.1 (already the named destination) +
     `design/15` §3.2.3 (needs amendment) + `design/04` §3.10 (hook payload
     growth). **Effort**: high (the single biggest item in our NNUE
     roadmap after the basic net lands). **Impact**: plausibly the single
     largest evaluation-strength lever available post-basic-NNUE, given it's
     a headline feature of the #2 SPCC engine — but it's explicitly a
     stretch goal, not part of the P13 v1 net, per our own doc's phrasing
     ("the destination, not the first step").

7. **Do not port**: the raw-transmute net file format (§7) — our CENN
   container is already strictly better (versioned, checksummed); the
   AVX-512VBMI2 `_mm512_maskz_compress_epi8` feature-extraction trick (§3,
   `psq.rs:80-103`) and the full vectorized ray-SWAR threat-delta machinery
   (§4.5) are both real engineering wins but are ISA-specific optimizations
   to revisit only after a scalar threat/AVX2-parity implementation exists
   and profiling says it's worth it — not day-one adoption targets.
