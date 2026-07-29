# ARCHON perf study #18 — the eager king-cross refresh goes lazy, and Stockfish's threat column stops being a lower bound

Date: 2026-07-29. Eighteenth commissioned study (17:
`perf-study-17-exact-counters-2026-07-29.md`, 16:
`perf-study-16-the-memset-class-closes-2026-07-29.md`). Raw artifacts and
instruments: `perf-artifacts-18/`. Baseline: `main` at **`8f556ee`** — studies
12–17 all merged and gated. Comparison target:
**`engines/bin/stockfish-eca43a97`**, plus a purpose-built `noinline` rebuild of
the same revision (§4.1).

---

## 0. HEADLINE

### 0.1 The frame, stated first

A GPU training run held the box for the whole session — available memory
between 0.9 and 3.0 GB, **swap 4095/4095 with zero free the entire time** — so
per the commission there is **no `check.sh`, no `CHECK_NNUE_GATE=1` and no
PGO build** here. Everything below is either the **thin-LTO probe frame**
(deltas) or the **callgrind frame** (attribution). Study 12 §0.3 still stands:
no probe delta may be quoted as a result. No wall-clock or NPS number is quoted
at all.

| | probe frame (thin LTO, 4 CGU, non-PGO, `-C target-cpu=native`) |
|---|---:|
| baseline (`main` @ `8f556ee`) | 6192.1 i/n |
| **shipped tree** | **6151.2 i/n** |
| **delta** | **−40.9 (−0.66%)** |
| branches | 663.0 → **647.9** per node (**−15.1, −2.3%**) |
| node count, every line | 5,001,562 |
| eval hash, every line | `5ca978e012da1bb0` |

**And a free confirmation of study 17's open threat-to-validity item 5.** That
study could not check whether its −114.5 and study 16's −29.2 composed, because
the two branches were unmerged and disjoint. They are both in this baseline:
**6335.8 − 114.5 − 29.2 = 6192.1**, and `null-1` measured 6192.1. Additive to
the last digit the instrument can print.

### 0.2 What landed

| # | change | probe Δ | where |
|---|---|---:|---|
| a1a | `ThreatPosition`-based from-scratch enumeration (`for_each_threat_at`), refresh still eager | **+7.1** | `sota_features.rs` |
| a1b | **the king-cross refresh becomes LAZY** — `Fold::Refresh` recorded at push, discharged in `compute` | **−44.2** | `sota_threats.rs` |
| a3 | `king_cross(dp) -> [bool; 2]` — one pass per `DirtyPiece` list instead of one per colour | −2.1 (**−8.8 br/node**) | `sota_threats.rs` |
| a4 | `#[cold] discharge_refresh` | −1.2 | `sota_threats.rs` |
| a6 | the D33 round — P-1 (fuse the walk's terminal `fold` load), P-4 (`[T; N]::map`), plus reviewer A's fixes | −0.5 (**−1.6 br/node**) | both |
| | **total, probe frame** | **−40.9** | |
| a2 | *(rejected, +22.7)* victim-type inner loop in `for_each_threat_at` | — | §6.1 |
| a5 | *(rejected, +0.1)* `#[cold]` on `pseudo_legal`'s non-Normal arm | — | §6.1b |

### 0.3 What this study is actually about

**(a) The commissioned target closed — and study 17's ~130 estimate is NOT
vindicated, though the first draft of §5 said it was.** Study 17 §11 item 1
sized the eager king-cross refresh at **~130 i/n**, on a callgrind measurement
of 222.7 and the model "only 41.5% of nodes evaluate, so ~58% is thrown away".
This study removes **−138.8 in the callgrind frame and −40.9 in the probe
frame**, and the temptation to report "dead on, then a 3.2× frame effect" is
exactly what §5 now retracts: **~84 of that −138.8 is one instrument artifact
and one codegen difference**, not work the shipped build stops doing. §5.2.

The model itself also needed correcting, in a way that generalises: **the saving
is 41.0%, not 58.5%, because a crossing ply that is never evaluated ITSELF is
still refreshed if any DESCENDANT evaluates.** The obligation propagates down
the subtree.

**(b) Stockfish's threat-generation column is no longer a lower bound.** Study
17 §4.4 had to publish a BRACKET — "+250 to +400 i/n, 1.30×–1.59×" — because
one SF symbol, `update_accumulator_refresh_cache`, contained an inlined
`FullThreats::append_active_indices` that "no regex can split", leaving
`0 ≤ X ≤ 433.4`. Its §11 item 1b named the fix and called it the first thing
study 18 should do. Done, in 90 seconds of build time: **X = 85.5**. §4.

**(c) With X measured, threat generation is 1.500× and it is still the largest
addressable block.** On an exhaustive bucketing over every symbol in both
engines, one frame, same 50 positions, same node budget: threat generation was
**ARCHON 1004.9 vs SF 598.1 — 1.680×, +406.8**, and is now **897.6 vs 598.1 —
1.500×, +299.5**. §4.3, including why that pair of ratios is *not* directly
comparable to study 17 §4.4's "1.30×–1.59×" bracket (three different
bucketings — D33 reviewer C, N-5).

**(d) One block that looks enormous in the callgrind frame is not measurable
there at all**, and saying so is the point. `acc fold` reads ARCHON 7850.4 vs SF
1926.9. **That is 4.07× and it is an artifact** — study 17 §6.2's runtime-
dispatch trap. Do not spend against it. §7.

---

## 1. METHOD

Studies 8–17's rules were followed.

### 1.1 NULL FIRST — three legs, run before any arm was written to disk

**Null A — rebuild determinism.** Same source, forced recompile:

```
null-1           30,970,000,227   6192.1 i/n
null-2-rebuild   30,970,000,964   6192.1 i/n      737 apart, ±0.00015 i/n
```

**Null B — SOURCE PERTURBATION, structurally inert.** Study 14's null failed
informatively (+86.4) when a "neutral" re-spelling was not, and study 16's fix
was to prefer a reborrow. Null B here is a reborrow of a reference parameter at
**exactly the three sites arm 1 was about to change** — `let board = &*board;`
in `for_each_threat`, `let dp = &*dp;` in `apply_at`, and an SSA-identical
rebinding of `ply` in `compute`:

```
n0-NULLB-reborrow 30,970,001,099  6192.1 i/n      872 from null-1, ±0.00017 i/n
```

**Largest null: 0.00017 i/n.** The smallest arm this study keeps (a4, −1.2) is
**7,000×** it; the smallest it rejects (a2, +22.7) is 130,000×. Nothing here is
noise in either direction.

The instrument's ability to answer the question is therefore not in doubt, which
is what rule 1 is a gate for.

### 1.2 Instruments

* **`probe18.sh`** — screening, inherited from `probe17.sh`, new target dir and
  output file only. Thin LTO / 4 CGU, `-C target-cpu=native`, non-PGO, fixed
  5,001,562-node tree; prints `instructions:u`, `branches:u`, the wl50 eval hash
  and the node count on every line. Every build inside its own
  `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0` cgroup **and**
  through `tools/oomlast.sh`. Build-and-measure cycle: **16 s**.
* **`cg18.sh` + `cgsum18.py` + `blocks18.py`** — study 17's exact per-function
  counters, unchanged except for paths.
* **`mut18.sh` — NEW.** Applies a mutation, builds, runs the fast `sota` tier,
  records which test caught it, reverts. §3.3, and it shipped a bug of its own
  worth reading (§3.4).
* **the `noinline` Stockfish rebuild** — §4.1. One line on top of
  `sfbuild17.sh`'s existing source export.

### 1.3 The box, honestly

Available memory 0.9–3.0 GB, swap exhausted throughout, one build at a time,
`free -m`'s SWAP line read before each. `probe18.sh` refuses below 2600 MB and
did so twice; those turns were spent reading `eca43a97` instead. **Nothing was
OOM-killed and the trainer never noticed.** Everything quoted rides on
`instructions:u` over a fixed-node tree or on deterministic callgrind counts,
both contention-immune.

---

## 2. THE TARGET, AND WHY IT WAS EAGER

`ThreatContext::apply_at` runs **once per made move**. Before this study, it did
this:

```
snapshot the post-move ThreatPosition into the child entry
for each perspective:
    if that perspective's king crossed the d/e mirror:
        refresh_threat_into(&mut child.acc[p],  weights,      board, p)   // 1280 lanes
        child.psqt[p] = refresh_psqt_threat(psqt_weights, board, p)
        child.computed[p] = true
    else:
        child.computed[p] = false
```

FullThreats indices are **king-oriented** — `orient_threats(ksq)` is `0` for a
king on files a–d and `7` for e–h — so when a perspective's own king crosses
that boundary every one of its ~80 active feature indices is re-oriented and the
parent's accumulator is worthless. There is no delta; the whole thing must be
rebuilt.

**It was eager for exactly one reason, and it is a plumbing reason, not a design
one:** `refresh_threat_into` took a `&Board`, and by the time a lazy fold would
want it, the live board has been unmade and re-made by a sibling. Everything
else in this module has been lazy since perf study 3.

The observation that unlocks it (study 17 §11 item 1, written out but not
attempted there because it needs the NNUE gate and would have collided with
study 16's then-unmerged branch — both now merged): **`apply_at` already stores
a `ThreatPosition` snapshot in the same entry, and that struct carries exactly
what the from-scratch enumeration reads.** `for_each_threat` wants
`king_square(p)`, `all_occupancy()`, `pieces(c, pt)` and `piece_on(to)`;
`ThreatPosition` carries `kings`, `occ`, `by_type`, `by_color`. Nothing is
missing.

---

## 3. WHAT LANDED

### 3.1 The three-state fold marker

`ThreatEntry::computed: [bool; 2]` becomes `fold: [Fold; 2]`:

```rust
enum Fold {
    Done,        // acc[p]/psqt[p] already hold this ply's value
    FromParent,  // fold the parent forward through this ply's dp/position
    Refresh,     // this ply's value does NOT come from its parent: rebuild
                 // from scratch out of `position`
}
```

`Refresh` **terminates `compute`'s ancestor walk exactly as `Done` does** —
both mean "this ply's value is not derived from its parent" — and is then
discharged in place, out of the ply's own snapshot. That is the whole
mechanism; it is four lines in the walk and one `#[cold]` helper.

The invariant it rests on, stated so a reviewer can attack it: *after the walk,
every ply strictly inside `(anc[p], ply]` is `FromParent` for `p`*, because
`anc[p]` is by construction the nearest ply ≤ `ply` that is not `FromParent`.
So the child loop only ever folds foldable plies, and a `Refresh` can never be
skipped over. The walk always terminates because `refresh()` sets
`stack[0].fold = [Done; 2]` and `apply_at` only ever writes `child = ply + 1 ≥ 1`.

### 3.2 One definition of the feature set, still

The thing that would make this change dangerous is two enumerations that can
drift. There are not two. `ThreatPosition` moved from `sota_threats.rs` to
`sota_features.rs` — it is the position view the FullThreats feature space
reads, not an internal of the incremental observer — and:

```
for_each_threat(&Board, p, f)      = for_each_threat_at(&ThreatPosition::from_board(board), p, f)
refresh_threat_into(&Board, …)     = refresh_threat_into_at(&ThreatPosition::from_board(board), …)
refresh_psqt_threat(&Board, …)     = refresh_psqt_threat_at(&ThreatPosition::from_board(board), …)
```

The `&Board` forms are thin wrappers kept for the root refresh and for the
`cfg(debug_assertions)` oracle. There is one enumeration body.

**This is not free: a1a measured +7.1 i/n** — the snapshot's `piece_on` is five
type tests where a `Board`'s is one mailbox byte. It is the toll for the
mechanism and it is paid on the (now rarer) rebuild only. §6.1 records a failed
attempt to win it back.

### 3.3 Tests that can fail, and five mutations that prove it

Two tests, both for a shape **the eager form could not produce** (it marked the
crossing ply computed at push, so a fold could never reach it in the `Refresh`
state):

* `deferred_king_cross_as_an_unevaluated_ancestor_matches_refresh` — the
  crossing ply is pushed and never read, five ordinary plies are stacked on top
  of it, and the first read of the line must discharge the refresh at the
  crossing ply and *then* fold the children.
* `a_crossing_sibling_overwrites_a_folded_ply` — a non-crossing ply is folded
  and memoized, then overwritten at the same ply by a sibling that crosses.

Five mutations, each applied, built, run under the fast `sota` tier and reverted
(`mutations.txt`, patches in `scripts/m*.py`). **All five failed a test, and
both new tests are among the catchers every time:**

| mutation | what it breaks | also caught by |
|---|---|---|
| m1 never discharge a `Refresh` | children fold from a stale accumulator | `threats_replay_matches_refresh`, `search_sota_accumulator_stays_consistent_across_the_tree`, 6 more |
| m2 never record the obligation | a crossing ply folds from its parent | same |
| m3 let the walk step past a `Refresh` | the crossing ply is never computed | same |
| m4 a sibling overwrite leaves a stale `Done` | the previous sibling's accumulator is read | `sibling_overwrite_at_same_ply_matches_refresh` |
| m5 the rebuild reads the OTHER perspective's king | wrong mirror orientation | 7 tests |

m1 and m3 are **only reachable at all because the refresh is now deferred** —
they are new failure modes, and the new test is what localises them to this
module rather than to "the search disagrees with a fresh context".

m5 is the class the snapshot form makes newly possible: `pos.kings[p]` where
there used to be `board.king_square(p)`.

### 3.4 A harness bug worth publishing, because it is study 17 §6.2.1 one level down

`mut18.sh` first decided "did it compile?" by grepping the combined cargo output
for `^error`. **A failing test prints `error` lines of its own**, so the harness
reported `DID-NOT-COMPILE` for exactly the runs that worked — it inverted the
verdict on the only runs that matter, and it would have reported a clean
mutation panel as five compile failures and a study with no test coverage as one
with perfect coverage, depending on which way the author read it. It now reads
cargo's exit status from a separate `--no-run` build.

Study 17 §6.2.1's rule was **A PARSER IS A BUCKETING**. This is the same rule
for a *verdict*: **an oracle that shares a channel with the thing it judges is
not an oracle.**

---

## 4. TARGET 2 — SF's `X`, MEASURED

### 4.1 The rebuild

`FullThreats::append_active_indices`
(`eca43a97:src/nnue/features/full_threats.cpp:210`) is called unconditionally
from `update_accumulator_refresh_cache`
(`eca43a97:src/nnue/nnue_accumulator.cpp:577`) and GCC inlines it, so it has no
symbol and study 17 could only bracket its cost. One `__attribute__((noinline))`
on that definition, rebuilt at `ARCH=x86-64-avx2` from study 17's read-only
`git archive` export, re-run through `cg18.sh` at the same 3000-nodes × 50
positions:

| | study 17 (inlined) | study 18 (`noinline`) | |
|---|---:|---:|---|
| `update_accumulator_refresh_cache` self | 433.43 | **347.97** | −85.46 |
| `FullThreats::append_active_indices` self | *(no symbol)* | **92.83** @ 0.1513 calls/node, 613.5 i/call | |
| whole-binary total | 14963.5 i/n | 14971.0 i/n | **+7.5, +0.05%** |
| SF node count | 150,053 | 150,053 | identical |

Its inclusive cost equals its self cost (GCC has inlined its own callees into
it), so there is no hidden tail.

**`X = 85.5`** — what the enumeration cost while inlined. (92.8 is the same work
plus the call overhead the `noinline` introduced; the 7.4 difference is
essentially the whole-binary +7.5, which is the cross-check.)

Study 17's bracket was `0 ≤ X ≤ 433.4`, and it wrote its illustrative rows at
X = 100 and X = 150. **The truth is 85.5**, i.e. below the bottom of the range
it actually reasoned in, and 5.1× below the top of the range it published.

### 4.2 What that does to study 17 §4.4's arithmetic

Study 17's own inclusive-form table, with X filled in:

| | ARCHON | SF | ratio |
|---|---:|---:|---:|
| §4.1 as first taken | 1148.9 | 674.8 | 1.70× |
| − `ensure_threats` (correction (i), correct) | 1074.8 | 674.8 | 1.59× |
| **+ X = 85.5 (correction (ii), measured)** | **1074.8** | **760.3** | **1.414×** |

So study 17's §4.4 correction was right in direction on both halves and wrong in
magnitude on the second, and its published bracket (1.30×–1.59×) **contains**
1.414× — the corrected statement holds. What does *not* hold is the reasoning
inside it: it treated X = 150 as the pessimistic case and X = 433.4 as
"implausible", when the real answer was below both.

### 4.3 The auditable form — exhaustive bucketing

The §4.2 table mixes inclusive and self costs on both sides, which is why study
17 also built `blocks17.py`: bucket **every** symbol by regex, print the
unbucketed row, and check that the columns sum to each engine's own total. That
is the form a reader can audit. Both engines, one frame, 150,059 vs 150,053
nodes, `cg/blocks-base.txt` and `cg/blocks-s18.txt`:

| block | ARCHON base | **ARCHON HEAD** | SF | HEAD delta |
|---|---:|---:|---:|---:|
| **threat gen** | **1004.9** | **897.6** | **598.1** | **+299.5 (1.500×)** |
| search | 1208.5 | 1208.3 | 940.9 | +267.4 |
| movepick | 860.0 | 860.0 | 699.1 | +160.9 |
| make/unmake | 421.6 | 421.6 | 384.5 | +37.1 |
| board queries | 303.9 | 302.2 | 548.3 | −246.1 |
| movegen | 23.0 | 23.0 | 89.9 | −66.9 |
| histories | 254.7 | 254.7 | 162.2 | +92.4 |
| tt | 65.2 | 65.2 | 150.4 | −85.2 |
| libc | 293.4 | 263.0 | 653.8 | −390.8 |
| eval head | 2207.6 | 2207.6 | 1376.1 | +831.5 |
| acc fold | 7850.4 | 7850.4 | 1926.9 | **not measurable here — §7** |
| startup/io | 6483.5 | 6483.5 | 7260.3 | −776.7 |
| unbucketed | 234.4 | 235.3 | 136.1 | +99.3 |
| **TOTAL** | **21379.3** | **21240.5** | **14971.0** | |

`598.1` is exactly `update_piece_threats<true>` 229.01 + `<false>` 60.06 +
`append_changed_indices` 216.19 + `append_active_indices` 92.83 — the bucketing
closes to the cent, which is the whole point of doing it this way. Our column
closes too (reviewer C re-summed both independently).

**Two caveats on those ratios, both from D33 reviewer C, both taken:**

* **N-4 — SF's 598.1 carries its own `noinline` overhead.** 7.37 of
  `append_active_indices`'s 92.83 is the call/prologue the inlined build never
  pays (§4.1's whole-binary +7.5 is exactly that). Like-for-like SF is **590.7**
  and the ratios are **1.701× → 1.520×**, not 1.680× → 1.500×. The error runs in
  SF's favour so the reported gap is conservative — but the point of the rebuild
  was to stop quoting bounds, so both forms are printed.
* **N-5 — do NOT read "1.500× lands inside study 17's 1.30–1.59× bracket" as
  confirmation.** Three different bucketings are in play: §4.4 used SF 674.8 /
  ARCHON 1074.8 (hand-picked, mixed inclusive/self); study 17's own
  `blocks-w50.txt` used 505.3 / 1066.4; `blocks18.py` uses 598.1 / 1004.9. The
  containment is a coincidence of denominators. **Every ratio in this study is a
  `blocks18.py` self-cost ratio**; that is the one that sums to each engine's
  total, and it is the only one to carry forward.

The self-cost form leaves each engine's attack lookups outside the block on
*both* sides (ours in `board queries` and `unbucketed`'s `sliding_attack`, SF's
`both_attacks_bb` in `board queries`), which is like-for-like. The inclusive
form of §4.2 pulls SF's inlined lookups *into* its threat column and leaves ours
out, which is not. **Quote the self-cost bucketing.**

### 4.4 So: threat generation is +299.5 i/n at 1.500×, and it is still the largest addressable block

Excluding `acc fold` (§7) and `startup/io` (irrelevant to a game), the ranked
gaps are: **eval head +831.5**, **threat gen +299.5**, search +267.4, movepick
+160.9, histories +92.4, make/unmake +37.1. `eval head` is dominated by the
1280-vs-1024 accumulator width the commission excludes as a retrain; threat gen
is not.

**A fidelity result that reframes the whole block (D33 reviewer B, verified
against `eca43a97`): Stockfish's threat refresh was ALREADY lazy, and the eager
form this study deleted was the ARCHON deviation.** `AccumulatorStack::push`
(`nnue_accumulator.cpp:64-71`) does no accumulator work at all; everything
happens in `evaluate_side` (`:78-101`). SF also has **no Finny cache for
threats** — `AccumulatorCaches::Entry` (`nnue_accumulator.h:65-78`) is
piece-placement only, and `:538-539` says so in a comment — so it rebuilds the
threat half from scratch exactly as we now do.

And one place where **we are ahead of the reference by construction**: SF's
refresh trigger is `HalfKAv2_hm::requires_refresh`, i.e. **any own-king move**,
because its accumulator fuses HalfKA and threats into one vector. Its own
comment at `nnue_accumulator.cpp:112-113` concedes that "threat feature set
refreshes require a king move across the center, i.e., a subset of halfka
refreshes" — a saving it cannot take. Our split accumulators let `king_cross`
use the narrow d/e predicate. That is a real architectural advantage already in
the bank, and it is worth knowing before anyone proposes fusing the two.

## 5. RECONCILING THE TWO FRAMES — AND THE FIRST DRAFT OF THIS SECTION WAS WRONG

Study 17 §11 item 1 predicted **~130 i/n**. This study measures, on the same
tree:

| frame | delta |
|---|---:|
| callgrind (AVX2 `x86-64-v3`, 3000 nodes/position) | **−138.8 i/n** |
| probe (AVX-512 native, 100,000 nodes/position) | **−40.9 i/n** |

**The first draft of this section said "the estimate was right in its own frame
to 1%, and the callgrind frame is 3.2× optimistic because a shallow tree
evaluates more often per node".** That is a single-cause story, it is quotable,
and D33 reviewer C broke it. It is retracted. The real answer is three causes
with very different characters, and **only one of them is a frame effect**.

### 5.1 The per-symbol decomposition, corrected

`cg/summary-archon-base.txt` vs `cg/summary-archon-s18.txt`, 150,059 nodes both:

| symbol | baseline | HEAD | Δ | which arm |
|---|---|---|---:|---|
| `ThreatContext::apply_at` | 152.16 @ 1.0000 | **81.02 @ 1.0000** | **−71.1** | mostly **a3**, see below |
| `refresh_threat_into` → `_at` | 79.29 @ **0.0363** | 47.89 @ **0.0214** | **−31.4** | a1b |
| `__memset_avx2_unaligned_erms` | 141.31 @ 0.6391 | 111.78 @ 0.6259 | **−29.5** | a1b — **but see §5.2** |
| `ThreatContext::at_full` | 465.28 @ 0.4151 | 470.82 @ 0.4151 | **+5.5** | the toll, net of a4 + P-1 |
| `record_square_touch` | 199.06 @ 1.0644 | 199.06 @ 1.0644 | 0.0 | untouched |
| `__memcpy_avx_unaligned_erms` | 136.83 | 135.94 | −0.9 | |
| | | **sum** | **−127.4** of −138.8 | |

`discharge_refresh` appears as a call edge at **0.0208 calls/node, 117.7 i/n
inclusive** — the whole deferred rebuild, priced in one place.

### 5.2 THREE CAUSES, NOT ONE — and two of them are not "the arm working"

**(i) `apply_at`'s −71.1 is dominated by arm a3, which the probe frame prices at
−2.1.** Verified here independently of the reviewer, from
`callgrind_annotate`'s per-file attribution:

```
base   src/types.rs:…::apply_at    8,148,311 Ir  (= 54.30 i/n)
HEAD   src/types.rs:…::apply_at    absent        (=  0.00 i/n)
```

That line is `Piece`'s **derived `PartialEq`** — `pc == king` inside the old
`king_crossed`, run `2 × (|subs| + |adds|)` times per made move. `king_cross`
compares `pc.piece_type` instead and runs the lists once. `king_cross` has **no
symbol** (`nm` confirms it is inlined into `apply_at`), so the work did not move
to another bucket; `apply_at`'s own self cost genuinely fell.

So one arm reads **−54.3 in the callgrind frame and −2.1 in the probe frame — a
26× disagreement**, on a function called exactly 1.0000 times per node in both.
It is not an eval-rate effect; there is no eval rate in it. It is a **codegen
difference between the two builds** (`-C target-cpu=x86-64-v3` + line tables vs
`-C target-cpu=native`), plus the ordinary fuzziness of line attribution in an
optimised build.

My §5.1's first draft instead claimed the −71.1 was "the eager refresh leaving,
0.0149 × ~2200 = 32.8 of it, and ~36 of `apply_at`'s own register pressure".
**Both halves were wrong**: `refresh_threat_into` is a *separate symbol*, so its
2,182 i/call was never inside `apply_at`'s self cost and the table already lists
that saving as its own −31.4 row — the draft double-counted it.

**(ii) The `__memset` −29.5 is an INSTRUMENT ARTIFACT and is worth ≈ 0 in the
shipped frame.** The removed calls price at `29.53 / 0.0132 = 2,237 Ir each`,
for a `dst.fill(0)` of **2,560 bytes** — **1.14 bytes per counted instruction**.
That is glibc's `rep stosb` ERMS path, which **callgrind counts once per byte**
while `perf stat -e instructions:u` retires the whole `rep stosb` as ~one
instruction. The work really is eliminated; the *count* is an artifact of the
instrument, and it does not appear in the −40.9.

**(iii) What is left is the arm, and it is the smallest of the three numbers.**
Strip (i) and (ii) from −138.8 and about **−55** remains: the refresh call rate
(−31.4), `apply_at`'s genuine share (~−17), the `at_full` toll (+5.5), and
change. The probe says the whole study is −40.9. Those are compatible without
any scalar conversion.

**So: is study 17's ~130 vindicated?** No. It agreed with a number that is
~84 i/n artifact and codegen. **The agreement was a coincidence**, and saying so
is the point of §12's rule 3 rather than an embarrassment to bury.

### 5.2b THE MODEL CORRECTION — 41.0%, not 58.5%, and this generalises

Study 17's item 1 said: "It runs at PUSH time, i.e. on every made move; only
**41.5%** of nodes ever evaluate, so **~58%** of it is thrown away."

Measured exactly, and this one is NOT frame-dependent — it is a call-rate ratio
inside one profile: the refresh call rate falls **0.0363 → 0.0214**, a factor of
**0.590**. **41.0% eliminated, not 58.5%.** (Call-edge cross-check, reviewer C:
base 0.0357 from `apply_at` + 0.0007 from the root refresh = 0.0364; HEAD 0.0208
from `discharge_refresh` + 0.0007 = 0.0215.)

The model is wrong in a way worth writing down because it applies to every
future laziness argument on this stack: **a crossing ply that is never evaluated
itself is still refreshed if any DESCENDANT evaluates**, because `compute` walks
*back* to it from the descendant. The obligation propagates down the subtree.
Only a crossing whose **entire subtree** cuts off is saved.

So the correct form of "defer it and the nodes that never evaluate stop paying"
is **"…and the SUBTREES that never evaluate stop paying"**, and the two differ
by 17.5 percentage points here. Any item ranked on the node-level form is
over-ranked.

### 5.2c Branches — and the integer decomposition, re-run per block on BOTH engines

Probe frame, exact per node: **663.0 → 647.9, −15.1 (−2.3%)**. Of that, a3's
one-pass `king_cross` alone is **−8.8**, far more than its −2.1 instructions
would suggest and the reason it stays; the D33 round added another −1.6.

Study 17 §6.2.1 left this as a bracket ("between 67 and 94 of the +149.6 integer
branch gap is threat generation… the true share is 45–63%") because the same
unsplittable Stockfish symbol carried 27.0 branches of mixed threat and fold.
**The `noinline` rebuild splits that too**, so the branch table has no bracket
left either. `--branch-sim=yes` on all three binaries, exact conditional
branches per node, SIMD kernels and start-up excluded (§7):

| block | ARCHON base | **ARCHON HEAD** | SF | HEAD delta |
|---|---:|---:|---:|---:|
| **threat gen** | **108.8** | **97.0** | **29.3** | **+67.7** |
| movepick | 105.9 | 105.9 | 67.1 | +38.8 |
| search body | 88.5 | 88.5 | 71.6 | +16.9 |
| make/unmake | 35.8 | 35.8 | 23.7 | +12.1 |
| board queries | 34.2 | 34.1 | 29.6 | +4.4 |
| histories | 22.1 | 22.1 | 12.4 | +9.6 |
| movegen | 3.8 | 3.8 | 8.3 | −4.5 |
| tt | 3.0 | 3.0 | 8.3 | −5.3 |
| SF refresh-cache (HalfKA half only now) | 0.0 | 0.0 | 20.2 | −20.2 |
| other | 48.2 | 48.2 | 17.5 | +30.6 |
| **TOTAL, integer code** | **450.4** | **438.4** | **288.1** | **+150.3** |

**Threat generation is 45.0% of the whole surviving integer branch gap** —
exactly, not "45–63%" — and this study took 11.8 branches/node out of it. It is
the same block the instruction table finds, measured with a completely
independent statistic, on both engines, with no unsplittable symbol left on
either side.

### 5.3 THE RULE THIS PAYS FOR — a frame ratio is not a scalar

Study 17 §4.3 said callgrind i/n must not be quoted as headroom because the
shallow tree inflates eval-coupled symbols. True, and insufficient. Here the
two frames disagree by **26× on one arm, ~1× on another, and ∞ on a third**
(the `memset` row does not exist in the probe frame at all). **You cannot divide
a callgrind delta by anything to get a probe delta.** Use callgrind for
*attribution* — which function, how many calls, how many instructions per call —
and the probe for *deltas*, and never convert between them.

## 6. WHAT DID NOT LAND

### 6.1 The victim-type inner loop — +22.7, and it is the more faithful spelling

`for_each_threat_at` filters victims with three hand-written unions
(`pawn_targets = P|N|R`, `minor_slider_targets = …|B`, `queen_targets = …|Q`)
and then asks `pos.piece_on(to)` — five type tests — for each hit. The
incremental observer next door does the opposite: `record_targets` iterates one
victim type at a time off `KEPT_VICTIMS`, so the victim's type is a loop
constant and only its colour has to be looked up.

Bringing that shape to the from-scratch side is attractive for three reasons and
one of them is not speed:

* `piece_on`'s five tests become `color_on`'s one;
* the `&Board` form's `board.piece_on(to).expect("target mask ⇒ a piece")`
  panic edge disappears entirely; and
* **both sides of the "incremental == refresh" oracle would then read `MAP`**,
  via `KEPT_VICTIMS`, instead of one side reading `MAP` and the other reading
  three unions a human has to keep in agreement with it. (They do agree today —
  verified entry by entry against `MAP` at `sota_features.rs:189`: pawn {P,N,R},
  bishop/rook {P,N,B,R}, knight/queen {P,N,B,R,Q}, king ∅.)

**It measured +22.7 i/n and was rejected.** Node- and hash-identical, so it is a
pure spelling result: five `att & typed(vt)` intersections per attacker cost
more than `piece_on` does on the few squares an attacker actually hits. A queen
attacking six pieces pays five intersections to save six five-test lookups.

Recorded here at length so study 19 does not re-derive it, and because the
fidelity argument is good enough that someone will want to.

*(Caveat, flagged rather than resolved: the helper `emit_targets` took SEVEN
arguments, one over the SysV six-register limit study 17 §5.1c measured at −14.2
on its own. It is `#[inline]`, so the limit should not bind, but "should not"
is not "did not". A re-try under six arguments is in §9 item 4.)*

### 6.1b The `#[cold]` sweep is now two-thirds CLOSED, by measurement

Study 17 §11 item 4 said "the `#[cold]` sweep is not finished" and named three
sites: `movepick`'s scoring loops, `see_ge`'s non-Normal short circuit, and
`pseudo_legal`'s non-Normal branch "(which allocates a 520-byte `MoveList`
frame)". Two of the three are now dead:

* **`see_ge`'s arm is already nothing.** `board.rs:1950-1952` is
  `if m.kind() != MoveKind::Normal { return 0 >= threshold; }` — a two-
  instruction terminal return with no locals, no table, no register pressure.
  There is nothing to outline. Item 4 named it from the ranking table, not from
  the source.
* **`pseudo_legal`'s arm measured +0.1** (`a5-pseudolegal-special-cold`, node-
  and hash-identical). It is the textbook §5 shape — a whole arm of the
  function's own top-level `MoveKind` dispatch, caller holds both arguments,
  and it owns the only `MoveList` in the function — and it bought nothing. The
  site runs at ~0.33 calls/node (movepick's TT-move validation,
  `movepick.rs:331,340`), and LLVM had evidently already sunk the frame.

That leaves `movepick`'s scoring loops as the only untested site of the three.
**The §5 rule now has n = 2 in the negative direction** (u3's nested block,
which cost +64.0, and this one, which cost nothing): satisfying the dataflow
condition is necessary, not sufficient. The arm also has to be *expensive
enough inline to have been setting something*.

### 6.2 Not attempted, and why

* **Deriving the child `ThreatPosition` from the parent by XOR.** Study 15 arm 3
  measured **+106**, and its note names `split_at_mut` as the untried idiom. Not
  retried here: it also breaks at ply 0, whose `position` is never written by
  `refresh()`.
* **SF's shifted-bitboard pawn enumeration**
  (`shift<NORTH_EAST>(cPawns) & pawnTargets`, `from = to - attkDir`, and
  `pushers` computed once per colour — `eca43a97:src/nnue/features/full_threats.cpp:225-250`)
  against our per-pawn `pawn_attacks(c, from)` loop. It is a real shape
  divergence and it is genuinely cheaper. It was not attempted because after
  a1b this code runs at **0.0214 calls/node**, so even a 30% win on it is under
  15 i/n in the callgrind frame and under 5 in the probe frame. §9 item 6.

---

## 7. THE BLOCK THAT LOOKS LIKE THE WHOLE ANSWER AND CANNOT BE READ HERE

`acc fold` in §4.3 reads **ARCHON 7850.4 vs SF 1926.9, 4.07×**. It is the
largest number in the table by a factor of six and it must not be spent against,
because **the callgrind frame cannot see the code that ships**.

Our SIMD dispatch is *runtime* (`is_x86_feature_detected!`). Under valgrind,
whose CPUID hides AVX-512, we fall back to the AVX2 kernels; natively the same
binary runs AVX-512. Stockfish's AVX2 rebuild runs the same code both ways. That
asymmetry is study 17 §6.2's near-miss headline, and its signature is on record:
callgrind counted **1.63×** the instructions perf retired for ARCHON while
matching Stockfish **to 1.6%**.

So `i8_deltas_into` (3725.61) and `i16_deltas_into` (3072.73) are AVX2 fallback
costs against a 1280-lane accumulator, and SF's `apply_combined` (1422.31) is
its shipped AVX2 code. The ratio is not a fact about the engines.

**Study 17 rule 3 restated: a runtime-dispatched kernel is invisible to any
instrument that changes what the dispatch sees.** It is worth repeating because
this table is the most quotable object in the study and the wrong row is the
biggest one.

---

## 8. THE PGO PREDICTION — TWO-SIDED, WITH THE FALSIFIABLE PART NAMED

The probe frame overstated the PGO delta by **1.12–2.0×** in studies 12–15
(layout and `memcpy` changes) and by **2.17×** in study 17 (−114.5 probe →
−52.8 PGO, inlining-boundary changes).

**This study's arms are neither.** −37.1 of the −40.9 is *work elimination*:
41% of a ~2200-instruction rebuild simply does not happen. PGO cannot make
un-executed work cheaper, and the rebuild's cost is dominated by
`add_column`'s SIMD accumulation, which profile data barely touches. The two
small arms are different: a3 (−2.1) and a6's P-1 (−0.5) are loop/load fusion,
which should survive; a4 (−1.2) is an `#[inline(never)]` and is exactly the
class study 17 showed PGO can push through zero.

One thing §5.2 changes about this prediction, and it is a warning rather than a
hedge: the callgrind frame priced a3 at −54.3 where the probe prices it at −2.1.
If the same codegen sensitivity exists between thin LTO and fat-LTO PGO, a3's
contribution could move in either direction by more than its own size. It is
2.1 of 40.9, so it cannot flip the total.

**Predicted: −25 to −45 i/n in the PGO frame, most likely −33 to −42.**

**The falsifiable part is the SHRINK FACTOR, not the sign.** I predict
**probe ÷ PGO < 1.5×**, against 2.17× for study 17 and 1.12–2.0× for 12–15. If
the panel comes back at −20 or worse (a shrink of 2×+), then "eliminated work is
frame-robust in a way that re-spelled work is not" is **wrong**, and the most
likely reason would be that PGO inlines `refresh_threat_into_at` into the eager
site in the *baseline* — making the baseline cheaper than thin LTO measured it,
and the delta correspondingly smaller. That would be a finding about the whole
probe instrument, not about this arm, and it would be worth more than the 40 i/n.

I do **not** predict the sign can flip. a1b removes calls; there is no spelling
of "do not do the work" that PGO turns into a cost.

---

## 9. RANKED REMAINDER — handoff to study 19

**Closed by this study, with numbers:** study 17 item 1 (the eager king-cross
refresh — done, −40.9 probe / −138.8 callgrind, but read §5.2 before
quoting the second number); study 17 item 1b's first half
(**X = 85.5**, SF's threat column is no longer a lower bound); study 17's
threat-to-validity item 5 (s16 + s17 compose additively).

**Do not reopen** (still): make/unmake (0.970× inclusive), `refresh_check_info`,
`attacks::tables()`, `capture_stage`'s jump table for instructions (five
spellings), the branch ratio as a bounds-check story.

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **`at_full` is now the block's whole story: 470.82 i/n at 0.4151 calls/node — 1134 instructions per evaluated ply.** It inlines `compute` AND `resolve_both`, i.e. the ancestor walk, the per-event index resolution for both perspectives, the psqt delta, and the chunked empty-list copy. Nobody has ever split it. Stockfish's counterpart, `append_changed_indices`, is **227.9 i/call**, and its generator `update_piece_threats<true>` is **126.9 i/call against our `record_square_touch`'s 187.0** | **the residual +310.2 lives here and in `record_square_touch`** | **high — exact, both engines** | study 17 item 1b's SECOND half, still owed and now the top item: `#[inline(never)]` on `compute`, `resolve_both`, `build_deltas` and the fold, re-run `cg18.sh`, price the four stages. **Run it at the 30k budget, not 3k** (study 17 §4.3) |
| 2 | **`record_square_touch` did not move: 199.06 i/n at 1.0644 calls/node, 187.0 i/call**, against SF's `update_piece_threats<true>` at 126.9. It inlines `record_targets`, `ripple` and `record_outgoing`. **1.47× on the per-call cost of the same job** | ~60 i/n at parity, callgrind frame | high | it is one symbol at ~1.0 calls/node, so `dis18.sh` + a counter is enough to attribute it. Compare against `eca43a97:src/position.cpp`'s `update_piece_threats` shape directly |
| 3 | **Study 17 item 3c (`ripple`'s cancelling pairs) is CAPPED, VERIFIED, and SCOPED — and the cap is small.** Three separate results. (i) Measured here with the existing `measure_fold_row_counts` oracle over 78,422 perspective-plies: resolved rows **6.05** per perspective-ply against a minimal symmetric difference of **5.85** — **overfold 1.034**, so at most **3.4% of folded rows** is on the table. (ii) D33 reviewer B verified SF's guard exactly: `move_piece` passes `noRaysContaining = from \| to` (`eca43a97:src/position.h:402-417`) and `update_piece_threats` skips the discovered emission iff `ray ⊇ {from, to}` (`:1201`); our `ripple` (`sota_threats.rs`) has no parameter that could express it. (iii) **But its SCOPE is far narrower than study 17 implied**: SF applies it only on `move_piece`, i.e. **quiet non-promotion moves plus the mover leg of en passant**. Captures go through `remove_piece` (default `-1ULL`, unguarded) + `swap_piece` (`ComputeRay=false`, so the ray scan never runs) — and that capture decomposition is one we already match exactly. Our own `resolve_into` doc records the cancel rate at **0.48% of events** (106.5 M → 0.51 M) | **≤ ~127 i/n callgrind; realistically ~0.5% of the fold** | **high — counted twice and verified against the reference** | **DOWN-RANKED. Do item 1 first.** If taken, it is one AND + CMP per slider at generation time, applied ONLY to `build_deltas`'s first branch and the ep mover leg — never to the capture branch, where SF has no guard either |
| 4 | **Re-try §6.1's victim-type loop under SIX arguments.** The rejected spelling passed seven to `emit_targets`; study 17 §5.1c measured the seventh argument alone at −14.2 (i.e. +14.2 to have it). +22.7 − 14.2 is still positive, so this is a long shot — but it is 20 seconds and the fidelity argument (both oracle sides reading `MAP`) is worth having | −14 of +22.7 at best | low | fold `perspective` and `ksq` into one struct, or hoist the `pos.typed(vt)` loads out of the `from` loop |
| 5 | ~~The branch decomposition has not been re-run per block.~~ **DONE — §5.2c.** Exact, both engines, no unsplittable symbol left: threat gen **108.8 → 97.0 vs SF 29.3**, and it is **45.0% of the whole +150.3 integer branch gap**. What is still open is only the SIMD half, which no instrument on this box can reach (§7) | closed for integer code | high | the SIMD half needs a COMPILE-TIME AVX2 build of ours that forces the dispatch instead of reading CPUID |
| 6 | **SF's shifted-bitboard pawn enumeration** (`eca43a97:src/nnue/features/full_threats.cpp:225-250`) vs our per-pawn loop, in `for_each_threat_at`. Real divergence, genuinely cheaper, now on a 0.0214-calls/node path | < 15 callgrind, < 5 probe | high on the shape, low on the value | only worth doing if item 1 finds the same shape inside `build_deltas`, where it runs at 1.06 calls/node |
| 7 | **`apply_at`'s remaining 83.0 i/call at 1.0 calls/node** — `ThreatPosition::from_board` (11 bitboard reads → an 80-byte store) plus the `DirtyPiece` copy plus `king_cross`. Study 15 arm 3's parent-XOR derivation measured **+106** and its own note says the codebase's idiom for that shape (`split_at_mut`) was never tried | 20–40 | medium | note it also breaks at ply 0, whose `position` is never written by `refresh()` — any retry must seed it |
| 8 | **Study 17 item 4's `#[cold]` sweep is two-thirds closed** (§6.1b): `see_ge`'s arm is a two-instruction terminal return with nothing to outline, and `pseudo_legal`'s measured **+0.1**. Only `movepick`'s scoring loops remain untested — and `MovePicker::init_quiets` is **552.42 i/n at 0.1624 calls/node, 3401.3 instructions per call**, the largest single non-SIMD symbol outside the threat block | 0–40 | low, after two nulls | one `probe18.sh` cycle. But `init_quiets` at 3401 i/call deserves a `cg18.sh` split before anyone outlines anything inside it |
| 9 | Inherited and unchanged from study 17: item 3 (`unmake_move`'s 168-byte restore, +25.5), item 3b (SF has no en-passant arm in `legal`), item 6 (`6b782211` as a games question), item 7 (pack `Piece` into one byte), item 8 (`check.sh` clippies default features only), item 8b (`legal_special`'s unreachable `_` arm) | | | |

### 9.1 Three method rules this study hands over

1. **LAZINESS IS A SUBTREE PROPERTY, NOT A NODE PROPERTY.** §5.2. "Only 41.5% of
   nodes evaluate, so 58% of this is wasted" over-ranked the arm by 17.5 points,
   because a deferred obligation at ply *k* is still discharged if *any*
   descendant of *k* evaluates. Rank against the fraction of *subtrees* that cut
   off entirely.
2. **AN ORACLE THAT SHARES A CHANNEL WITH THE THING IT JUDGES IS NOT AN ORACLE.**
   §3.4. `mut18.sh` read "did it compile?" off the same stream the test failures
   print to, and inverted its verdict on precisely the runs that mattered.
   Study 17's "a parser is a bucketing", one level up.
3. **AN ESTIMATE CARRIES ITS FRAME.** §5. Study 17's ~130 was accurate to 1% in
   the callgrind frame and 3.2× optimistic in the shipped one. It even said so
   (§4.3). Any i/n handed forward in a ranking table should be stamped with the
   frame it was measured in, or it will be spent in the wrong currency.

---

## 10. WHAT IS OWED BEFORE THIS MERGES — THE PENDING CHECKLIST

Green **here**, on HEAD:

| check | result |
|---|---|
| `cargo test` (default; includes the fast perft tier + checked mini-tier) | **608 passed, 0 failed** |
| `cargo test --features sota,arch-d48x` | **695 passed, 0 failed, 8 ignored** |
| `cargo clippy --all-targets` | **0 warnings** |
| `cargo clippy --all-targets --features sota,arch-d48x` | **6 warnings, all pre-existing, all in `simple_halfka.rs`** (study 17 item 8's known pair; none in a file this study touches) |
| `cargo fmt --check` | **clean** |
| `cargo test --release --test perft_suite -- --ignored perft_full_suite` (**D24 ritual**) | **PASS** — 1 passed, 6.1 s |
| probe-tree node identity (5,001,562) + eval hash, every arm | **PASS** |
| the five-mutation panel | **all five caught** |

Deferred to the owner, in this order:

1. **`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=4 bash tools/check.sh` — MANDATORY.**
   This study is entirely under `src/eval/nnue/`, which the `Bench:` pin is
   blind to. Expect **all three pins UNCHANGED**: `Bench: 587683`, eval hash
   `5ca978e0…8b38`, d13 nodes `1922947`. Every arm is node-identical and
   eval-hash-identical, so **a moved pin is a bug in this study, not a new
   pin.**
2. **`cargo test --release -- --ignored perft_extended`** (~18 G nodes, < 4 min)
   — optional. The **full** tier is green here (table above); the extended tier
   is a release-and-movegen-rewrite gate and this study is neither, so it is
   listed for completeness only.
3. **`bash tools/build_pgo.sh sota,arch-d48x engines/bin/archon-s18
   engines/canonical/net.cenn`**, then the two-arm panel against
   `engines/bin/stockfish-eca43a97`
   (`pstat8.sh 5 300000 tools/nnue/calib/wl50.fens …`) — the deliverable frame,
   and the test of §8's **−25 … −45** prediction and its named falsifiable
   shrink-factor claim.

### 10.1 Node identity, argued rather than asserted

Two independent statements, and they cover different halves:

* **The wl50 eval hash** (`5ca978e012da1bb0`, unchanged on all nine probe lines)
  is a **from-scratch** forward pass on 50 positions. It pins the enumeration —
  which this study *did* change, by routing it through `ThreatPosition` — and it
  pins nothing about `compute`.
* **The node count** (5,001,562, unchanged on all nine lines including the
  rejected arm) is a 5 M-node search. Any divergence in the incremental
  accumulator changes a score, changes a cutoff and changes the tree. It pins
  `compute`; it is the statement that matters for a1b.

Plus the `cfg(debug_assertions)` oracle
(`SotaContext::assert_matches_refresh`) comparing incremental against
from-scratch at **every** evaluated node, which is what the whole `sota` debug
test suite runs under.

**This is evidence, not proof, for the shipped configuration**: the probe tree
is one 50-position search, and the gates in §10 are what replace "almost
certainly".

---

## 11. THREATS TO VALIDITY

1. **Nothing here is PGO-measured** (§0.1, §8). Machine constraint, not choice.
2. **The `acc fold` row of §4.3 is not measurable in this frame** and is the
   biggest number in the table (§7).
3. **THE PER-SYMBOL CALLGRIND DECOMPOSITION CANNOT BE CONVERTED INTO PROBE
   DELTAS.** §5.2, and this study got it wrong in its own first draft before the
   D33 round caught it. Two of the three largest rows in §5.1 are an ERMS
   counting artifact and a codegen difference; the frames disagree by 26× on one
   arm. Every i/n in §9's table is a callgrind-frame number and none of them is
   headroom.
4. **`X = 85.5` is measured on a `noinline` rebuild**, so it is the cost of the
   enumeration *plus* whatever GCC would have specialised away by inlining it
   into one call site — bounded by the +0.05% whole-binary move, but not zero.
   §4.3's N-4 note prints the like-for-like ratio alongside.
5. **The three published threat ratios come from three different bucketings**
   (§4.3, N-5). Only the `blocks18.py` self-cost pair carries forward.
6. **The +7.1 toll of a1a was measured against the EAGER refresh** (0.0357
   calls/node). On the shipped lazy path it runs at 0.0208, so the surviving
   toll is nearer +4. Not separately re-measured.
7. **The rejected arm a2 has an unresolved confound** — a seven-argument
   `#[inline]` helper (§6.1). Its verdict (+22.7) stands; its *explanation* is
   one experiment short, and D33 reviewer C's arithmetic says the stated
   mechanism should have cost far more than 22.7, so something else is also in
   play. The patch is preserved at
   `scripts/a2-victim-type-loops-REJECTED.py` so the spelling can be re-run.
8. **The `Fold::Refresh` path depends on the snapshot being the true post-move
   position** (D33 reviewer B). Today `apply_at` always receives a real
   `&Board`, so it is; unlike the fold, a wrong snapshot here has no telescoping
   identity that would make the error cancel. A future synthetic push would
   break it silently.
9. **Box state**: a GPU training run shared the box for the entire session and
   swap was exhausted throughout. No wall-clock or NPS number is quoted as a
   claim.
10. **The first `archon-s18` callgrind panel measured the a3 tree, not HEAD**
    (D33 reviewer C, N-1) — `discharge_refresh` had zero occurrences in it. It
    was re-run on HEAD and every number in §4.3 and §5.1 is from the re-run. The
    superseded panel is not in the artifacts.

## 12. ARTIFACTS (`perf-artifacts-18/`)

* `probes-fast.txt` — all nine arms with eval hash, node count and branch count,
  including all three nulls and the rejected arm.
* `mutations.txt` + `scripts/m1..m5.py` — the five-mutation panel.
* `cg/summary-archon-base.txt`, `cg/summary-archon-s18.txt` — exact per-function
  self-Ir, calls/node and i/call, before and after, same frame.
* `cg/summary-sf-noinline.txt` — the Stockfish profile with
  `append_active_indices` split out. **This is the artifact that closes study 17
  item 1b's first half.**
* `cg/blocks-base.txt`, `cg/blocks-s18.txt` — the exhaustive bucketing both
  engines. Read §7 before quoting the `acc fold` row.
* `cg/callgrind-*.out` (+ `.log`) — the raw callgrind files.
* `scripts/probe18.sh`, `mut18.sh` (new), `cg18.sh`, `cgsum18.py`,
  `blocks18.py`, `brint18.py`, `cgbr18.sh`, `cgbuild18.sh`, `probeany18.sh`,
  `dis18.sh` — instruments. Everything except `mut18.sh` is study 17's, path-
  adjusted.

The Stockfish `noinline` build tree is at `/home/nick/archon-sf17-eca43a97`
(study 17 left it on purpose; this study patched one line of
`src/nnue/features/full_threats.cpp` and rebuilt). `src/stockfish-plain` beside
it is the un-patched binary. The ARCHON AVX2 tree is `target-cg18/`.
