# Perf study 71 — the TT-less deficit, re-priced; and the one ordering signal we did not have

**Target: study 70's largest untouched block** — 46 % of NonPV cutoffs arrive
with no usable TT move, their first-move cutoff rate is 14.9 pp worse, and they
own ~70 % of the engine's wasted cutoff work.

Everything below is reproducible from this tree:

```
cargo build --release --features sota,arch-d48x,census70,arm70   # the instrument
tools/perf71/cscreen71.py 'qsee=49152' ...     # census screen, null panel first
tools/perf70/screen70.py  'qsee=49152' ...     # paired node screen
tools/perf70/ipn70.sh A B 3 11 tools/perf70/uho300.fens   # instructions per node
```

---

## 0. The one-paragraph version

The census could not say which repair applied, because three different failures
share one symptom, so the first thing built was the split. It says: of the
757,789 TT-less NonPV node entries, **42.6 % are hard probe misses, 57.4 % are
entries that exist and carry no move, and 0.001 % are validation failures** —
and **half the moveless half was written by a main-search fail low**, which
knows perfectly well which move scored highest and stores `Move::NULL` anyway.
Storing that move is the obvious fix; measured against its own pruning control
it is **worth nothing** (−0.06 pp of `fhf`, +2.3 % nodes). The pruning control
itself — letting IIR fire on UPPER-bound entries — cuts the tree **27.7 %** and
**loses 24.17 ± 15.11 Elo over 792 games**. Then the deficit itself was nulled
for the first time: ordering the SAME nodes without their TT move costs
**9.6 pp, not 14.9** — a third of study 70's headline is population, not
ordering. What did land is a term the deficit's own stage census pointed at: a
**SEE penalty on quiet ordering**, the one signal Stockfish's `score<QUIETS>`
does not have, clamped so it can only reorder. TT-less `fhf` **0.7632 →
0.7721**, and **+10.44 ± 9.73 Elo over 1,764 games** where the unclamped version
of the same term — better on every proxy — lost 5.35 ± 9.26 over 1,948.

---

## 1. The instrument: why a node arrives TT-less

Study 70 measured the deficit and stopped, having killed two hypotheses about
it. It could not rank the repairs, because "no usable TT move" is three
different events:

| | NonPV | share | what would fix it |
|---|---|---|---|
| probe MISS — no entry at all | 322,702 | **42.58 %** | TT capacity / replacement |
| entry present, **move is NULL** | 435,079 | **57.41 %** | **store policy** |
| move present, `pseudo_legal` rejects it | 8 | 0.00 % | nothing; not a problem |

`census70::TT_WHY`, uho300 @ d11, 757,789 TT-less NonPV node entries. The third
row is the cheapest finding in the study: key-collision and stale-entry
validation is **not** a source of the deficit, at eight events in 4.57 M nodes.

The moveless majority splits again, by the depth its entry was stored at, which
names the store site that wrote it:

| moveless entry written by | NonPV | share |
|---|---|---|
| an eval-only cache write (`DEPTH_UNSEARCHED`) | 119,935 | 27.57 % |
| a quiescence fail low | 95,234 | 21.89 % |
| a main-search fail low, depth 1–3 | 149,958 | 34.47 % |
| a main-search fail low, depth ≥ 4 | 69,952 | 16.08 % |

**Half the moveless population is a main-search fail low** — 220 k node entries,
17.7 % of every NonPV node in the run. And the composition is depth-dependent in
the direction that matters: at depth 1 the TT-less population is 71 % hard miss,
but from depth 3 up it is 72–85 % **moveless**, and depths 3–5 are where ~49 % of
the engine's excess move-loop iterations live.

## 2. The obvious fix, and its pruning control

At a fail-low node no move raised alpha, so `best_move` is NULL and the argmax
over the searched moves is thrown away. `TT_FAILLOW_MOVE` stores it. Neither
Stockfish `eca43a97` (`search.cpp:1782`) nor Reckless (`search.rs:945`) does.

It cannot be screened alone. Four sites read `tt_data.mv == Move::NULL` as
"this node is unresolved, reduce or prune it harder" — internal iterative
reduction, reverse futility's `ttMove == NULL || ttCapture` gate,
`RED_CUTNODE_NO_TTMOVE` and `RED_FULLDEPTH_NO_TTMOVE` — so handing ~20 % of
NonPV nodes a move they did not have flips all four. That is study 70's standing
finding arriving in a new costume, and it cost that study two retracted headline
arms. So `TT_UPPER_PRUNE_BLIND` was built as a **separate** knob (a bitmask over
the four sites) and the arm was screened as a 2×2.

uho1000 @ d10, null panel ±0.03 pp on `fhf`, ±2.5 % on nodes:

| arm | store | prune-blind | Δ fhf NonPV | Δ nodes |
|---|---|---|---|---|
| null (jitter=1, 3 seeds) | – | – | −0.03 … +0.03 | +2.0 … +2.9 % |
| `flmove=1` | ✓ | | **+0.08** | +3.9 % |
| `ttup=1` (the control) | | ✓ | +0.83 | **−28.9 %** |
| `flmove=1,ttup=1` | ✓ | ✓ | +0.77 | −27.2 % |

**The store is a null and slightly negative.** Against the `ttup=1` control it
reads **−0.06 pp of `fhf` and +2.3 % nodes**. The `fhf_no` +3.0 / `fhf_yes`
−3.0 pattern that looks like a win is pure re-labelling: nodes move from the
"no TT move" bucket to the "has one" bucket and cut first about as often as they
did before, i.e. the fail-low argmax is barely better than the static ordering
that produced it.

**Mechanism, stated so the door stays shut.** At a NonPV all-node with a zero
window every child returns an upper bound from a reduced search, and the moves
that survive the §3.5 pruning cascade to *get* a value are the ones the static
ordering already ranked first. The argmax over that filtered, reduced set
carries almost no information the next visit did not already have.

### 2.1 The control was a −24 Elo arm in its own right, and that is a result

`ttup=1` is not a null — it is a **27.7 %** tree reduction, and the decomposition
says the whole thing is one site:

| bitmask | site | Δ nodes |
|---|---|---|
| 1 | IIR | **−27.7 %** |
| 2 | RFP's `ttMove == NULL \|\| ttCapture` gate | +0.3 % |
| 4 | `RED_CUTNODE_NO_TTMOVE` | +0.6 % |
| 8 | `RED_FULLDEPTH_NO_TTMOVE` | −0.05 % |
| 15 | all four | −30.0 % |

So: making internal iterative reduction fire whenever the entry is an UPPER
bound — "the last search of this position failed low, so its move is not
evidence the node is resolved" — cuts the tree by more than a quarter. It was
built as a PGO pair and sent to games immediately.

**REJECTED: −24.17 ± 15.11 Elo, 792 fixed-node games, LOS 0.08 %.** Our IIR is
already exact SF parity (`search.cpp:1030`, `!ss->followPV && !allNode && depth
>= 6 && !ttData.move`), and SF's source carries the comment `(*Scaler) Making
IIR more aggressive scales poorly`. This is that comment, measured on ARCHON.

## 3. The deficit itself, nulled — and it is 1.55× over-priced

Study 70 priced the block as the gap between `fhf(has a TT move)` = 0.9134 and
`fhf(no TT move)` = 0.7632. That subtraction assumes the two populations are
otherwise comparable. **They are not**: a node with a TT move is one the search
has already visited, and re-visited nodes may simply be easier.

`hidett=1` separates them. It hides the TT move from the **MovePicker only** —
every pruning, reduction and extension gate still reads `tt_data.mv`, so the
tree changes for exactly one reason: the ordering lost its best hint. The census
still splits by whether the node *had* a usable TT move.

| population | shipped | ordered without the TT move |
|---|---|---|
| NonPV, node HAD a TT move | **0.9134** | **0.8175** |
| NonPV, node had none | 0.7632 | 0.7722 |

**The TT move is worth 9.6 pp of ordering, not 14.9.** The other 5.4 pp is the
population: nodes that own a TT move cut first 0.8175 of the time *with no TT
hint at all*, against 0.7632 for nodes that never had one. A static orderer that
made TT-less nodes as orderable as TT-having ones would close 5.4 pp, not 15,
and the remaining 9.6 pp is what "having the right move" is worth — which is
what a TT move *is*.

**Study 70's headline is over-priced 1.55× as a static-ordering target.** Both
halves are real; only one of them is reachable without a TT hit.

## 4. Where the reachable half actually is

With store policy dead, the whole 757,789-entry TT-less population depends on
static ordering. The census was extended to say which stage emits the cutting
move, at which index, and — for quiet cutoffs — how many captures preceded it
and where it ranked among the quiets.

**TT-less NonPV cutoffs by stage and index** (n = 295,793):

| cutting move at index | n | GoodCapture | GoodQuiet | Bad* |
|---|---|---|---|---|
| 1 | 225,758 | 56.1 % | 43.9 % | 0.0 % |
| 2 | 37,898 | 21.1 % | **78.8 %** | 0.1 % |
| 3 | 12,666 | 7.4 % | **90.7 %** | 1.9 % |
| 4+ | 19,471 | 0.5 % | 63.0 % | 36.5 % |

When the first move fails, the move that actually cuts is a **quiet**, three
times out of four. Two hypotheses follow, and the census separates them:

| for TT-less quiet cutoffs (n = 156,444) | |
|---|---|
| captures emitted before the cutting quiet = 0 | **85.5 %** |
| the cutting quiet was the top-scored quiet (`qrank == 1`) | **71.5 %** |
| `qrank ≥ 2` — the quiet SCORING put the wrong quiet first | **28.5 %** (44,637) |

So the capture/quiet interleave is **not** the lever — most TT-less cut nodes
have no good captures at all (mean 0.18 before the cutting quiet). The lever is
the quiet score itself, at 44,637 cutoffs, 15.1 % of the TT-less population.

## 5. What landed: a SEE term on quiet ordering

Our quiet scorer is an exact transcription of Stockfish `eca43a97`'s
`score<QUIETS>` plus the D30 offense term, so "add a term SF has" was not
available. What SF does **not** have is any SEE term at all. Its nearest relative
is the threat tier

```
PieceValue[pt] * QUIET_THREAT_FACTOR * (attacked-at-from − attacked-at-to)
```

and that tier knows only **strictly cheaper** attackers, is blind to whether the
destination is defended, and for a **pawn** is literally `Bitboard::EMPTY`.

```rust
if QUIET_SEE_PENALTY() != 0
    && value > GOOD_QUIET_THRESHOLD()
    && all_threats.contains(m.to())
    && !board.see_ge(m, 0)
{
    value = (value - QUIET_SEE_PENALTY()).max(GOOD_QUIET_THRESHOLD() + 1);
}
```

Both guards are free. `all_threats` is the enemy attack map `init_quiets`
already built for the tiers, and a quiet landing on a square no enemy attacks
passes `see_ge(m, 0)` trivially. The block is placed **last**, after the threat
and offense terms, so the second guard reads the finished score — which is not
just cheaper but the non-redundant order, since the tier has already sunk
"queen onto a rook-attacked square" below the threshold. Together: **0.093
`see_ge` calls per scored quiet**, down from 0.354 with the cheap guard alone.

### 5.1 The clamp is the whole change, and games said so

Unclamped, a penalty this size pushes the quiet into the **bad** pile, where it
is emitted after the bad captures and becomes a late-move-pruning candidate. The
term stops being ordering and becomes pruning — and the pruning version looks
**better on every proxy**:

| | Δ fhf NonPV | Δ fhf TT-less | Δ nodes (uho300 d11) | Δ instructions/node | Δ total instructions | **games vs the same PGO baseline** |
|---|---|---|---|---|---|---|
| unclamped, 32768, one guard | +0.42 | +0.71 | −0.4 % | +7.66 % | +8.06 % | **−3.71 ± 8.86** (2,156) |
| unclamped, 49152, both guards | +0.39 | +0.73 | **−6.34 %** | +3.43 % | **−3.13 %** | **−5.35 ± 9.26** (1,948) |
| **clamped, 49152, both guards** | **+0.50** | **+0.89** | −2.05 % | +3.99 % | +1.86 % | **+10.44 ± 9.73** (1,764, LOS 98.2 %) |

**Mechanism for the sign flip, and it is not a mystery**: the search *already*
prunes losing quiets by SEE in the move loop, at `-QSEE_MULT · lmr_depth²`
(`search/mod.rs:3930`, SF parity) — depth-scaled and permissive. The unclamped
ordering demotion is a second, harsher, **depth-blind** copy of the same rule,
and SEE is a static approximation that mis-prices pins, x-rays and sacrifices.
Clamped, the term cannot prune anything; it only reorders inside the good pile,
and it keeps the *larger* ordering gain.

This is the second instance in two studies of study 70's own warning —
**ordering quality is a proxy and it Goodharts** — arriving from the other
direction: there the metric kept improving while the search got worse; here the
node metric improved 3× more while the games got worse.

### 5.2 The controls

| control | reading | what it rules out |
|---|---|---|
| **size- and magnitude-matched RANDOM** — demote a pseudo-random 23 % of quiets (the share this predicate selects) by the same amount | **−1.17 pp fhf, +17.2 % nodes** | the gain is not "demoting a quarter of the quiets"; it is the predicate |
| **`see_ge(m, +100)`** — a predicate that fires on far more moves | −1.95 pp fhf, +13.0 % nodes | the threshold is not free either; 0 is the right one |
| **`qeq`** — widen the tier to EQUAL-or-cheaper attackers, no `see_ge` at all | +0.19 pp at best, **−0.59 at this magnitude** | the free approximation does not work: without knowing the destination is defended, "attacked by an equal piece" is a bad predicate. The SEE call earns the term. |
| **`qseept`** — pawns only / non-pawns only | +0.09 / +0.33 | both movers contribute; the pawn half exists because the tier is EMPTY there |
| **`qseedepth=2 / 3`** | +0.28 / +0.19 | the shallow nodes are where the value is; do not gate it |

### 5.3 The constant is not resolved, and is left to SPSA

Every point in 24576…98304 reads +0.43…+0.56 pp of `fhf` (14–18× its null). The
paired node metric ranks them differently on the two position sets and
differently again between its geometric mean and its sign test — that is the
node metric's chaos, not a curve, and the honest statement is a plateau. 49152
is inside it.

## 6. Games

`tools/perf70/chain70.sh` (study 70's segmented, resume-safe runner) with
`NODES=25000 BOOK=UHO_Lichess_4852_v1.epd HASH=16`, `CONCURRENCY=6`, both sides
PGO from the same source, `option.EvalFile` verified on the **live** fastchess
argv in every run. Baseline `engines/bin/s71-base` reproduces the pre-study
fingerprints exactly (wl50 d13 = 1,979,892; uho300 d11 = 4,565,705).

| arm | pairs | games | score | Elo | LOS |
|---|---|---|---|---|---|
| `ttup=1` (IIR on UPPER entries) | 396 | 792 | 46.53 % | **−24.17 ± 15.11** | 0.08 % |
| quiet SEE, unclamped 32768 | 1,078 | 2,156 | 49.47 % | −3.71 ± 8.86 | 20.6 % |
| quiet SEE, unclamped 49152 | 974 | 1,948 | 49.23 % | −5.35 ± 9.26 | 12.9 % |
| **quiet SEE, CLAMPED 49152 (shipped)** | **882** | **1,764** | **51.50 %** | **+10.44 ± 9.73** | **98.2 %** |

**The bar clears zero on the shipped arm** (+10.44 − 9.73 = +0.71) at 1,764
games with LOS 98.2 % — but this is a pooled Elo bar, not an accepted SPRT
verdict, and it is stated as the former. The chain `s71-q3chain` is still
accumulating; read the running total with

```
tools/perf70/pool_sprt.py engines/runs/s71-q3chain-*/sprt.out
```

**A caveat that the protocol cannot see, stated rather than buried.** These are
FIXED-NODE games, so they are blind to the term's +3.99 % instructions per node.
At study 41's measured instructions→wall-clock ratio (~2.6×) that is roughly
1.5 % of real time, ~2.6 Elo at this engine's ~120 Elo per speed doubling. The
fixed-node result is therefore an over-estimate by about that much, and a
real-TC confirmation on a quiet box is owed before this is treated as settled.

**Gate:** `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` exits 0.
Eval hash **`2e1ede95…79c4`, unchanged** — this is search, not eval. Both node
pins move, as a functional change must: `Bench:` 401733 → **300694**, NNUE d13
1979892 → **2045128**.

---

## 7. What this study did NOT close, with numbers

- **The hard-miss half of the deficit** — 322,702 NonPV node entries (42.6 %)
  whose probe finds nothing, 162 k of them at depth 1. At 15,219 nodes per
  position against 1.57 M TT entries the table is not under pressure, so these
  are genuine first visits and no store or replacement policy reaches them.
  Only static ordering does, and §3 caps what static ordering can be worth
  there at 5.4 pp.
- **The remaining 28.5 % of TT-less quiet cutoffs that are ordered wrong**
  (44,637 events). The SEE term takes some of them; the rest is the seven-plane
  history stack, whose component weights study 70 already screened (`mainw`,
  `threatw`, `offensew`, `c1w`, `qsortfull` — all null or negative).
- **The LMR re-search block.** Study 70 measured 3.64 % of NonPV scouts forcing
  a full-depth re-search at 21.4 % of the tree, 41.5 % of them at move two. This
  study did not touch it; the shipped change moves the rate −0.15 pp.
- **The eval-only moveless entries** (119,935, 27.6 % of the moveless
  population). These are positions that took an RFP/razoring/null-move cutoff
  and were never searched, so no move exists to store. Nothing here is
  recoverable by a store policy; it is a *pruning* population wearing an
  ordering symptom.

## 8. Reproducing

- `tools/perf71/cscreen71.py ARM...` — census screen: `fhf`, the TT split, the
  moveless share, with the null panel first. `BASE=qsee=0` A/Bs the landed
  change against its own predecessor inside one binary.
- `tools/perf70/screen70.py ARM...` — the paired node screen (sign test).
- `tools/perf70/ipn70.sh A B [rounds] [depth] [fens]` — instructions per node.
- `tools/perf70/chain70.sh CAND BASE tag segments rounds` — the segmented match.
- `census70` prints, per node kind: `ttwhy` (miss/moveless/unusable),
  `mlsrc` (which store site wrote the moveless entry), `srctt`/`idxtt`/`srcidx`
  (cutting stage by TT availability and index), `capsbefore`/`qrank`
  (the two causes of a late quiet cutoff), `quietsee` (the SEE-loss and
  SEE-call rates).
- `arm70` knobs added here: `flmove`, `qflmove`, `ttup`, `hidett`, `qsee`,
  `qseeg`, `qseegq`, `qseept`, `qseedepth`, `qseeclamp`, `qseepv`, `qseethr`,
  `qeq`, `qseerand`, `qseerate`.
