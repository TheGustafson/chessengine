# Perf study 34 — move ordering: what `init_quiets` does, and what the search ever uses

**Date:** 2026-07-30
**Scope (owned files):** `src/movepick.rs`, plus the study's own instrument
(`src/census34.rs`) and its hooks in `src/search/mod.rs`, `src/bench.rs`,
`src/lib.rs`, `Cargo.toml`. ARCHON-only — no Stockfish comparison was made or
needed.

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100k nodes, one thread, 16 MB TT, D48X net, features
`sota,arch-d48x`, `RUSTFLAGS=-C target-cpu=native`, the SHIPPED release
profile (fat LTO, **CODEGEN_UNITS = 1** — study 23 §2.2 showed `instructions:u`
is a lottery at CGU 4), pinned to core 2.

The study straddles study 32's merge, so two frames appear and each number says
which it is:

| tree | nodes | wl50 eval hash |
|---|---:|---|
| pre-rebase (study 23's frame, unchanged since) | 5,001,562 | `5ca978e012da1bb0` |
| post-rebase (study 32's i8 HalfKA landed) | 5,001,624 | `eb981b119bd66d6e` |

**Box conditions: NOT quiet.** Studies 32, 33 and 35 were live in sibling
worktrees. A sibling `perf` process was running inside the measurement window
(confirmed by `pgrep`). This is stated up front because it decided which
instrument this study is allowed to quote — see §6.

---

## 0. TL;DR

**Ordering QUALITY is not the lever; cost is.** 86.44% of all beta cutoffs land
on the first move tried (93.89% by move 2, 97.81% by move 4; mean move index at
cutoff 1.316). That is the answer to the question the brief posed, and it
points the whole study at cost.

**Three measurements nobody had:**

1. **The consumption census (§3).** `init_quiets` scores 22.57 quiets per call
   and the search ever *looks at* 15.59 of them and ever *emits* 7.41 — **32.9%
   of the scoring work is used.**
2. **The decomposition of `init_quiets` on the SHIPPED build (§4)**, which
   inlines four different mechanisms into one 6.59%-of-cycles symbol. This
   **corrects study 23's headline sort figure from 2.84% to 2.00%** of engine
   cycles, in exactly the direction that study's own caveat predicted.
3. **The mispredict localisation (§5).** `init_quiets` is the **second largest
   branch-mispredict source in the whole engine (13.75%)**, behind only
   `search::<NonPv>`; 61.52% of that is the sort, and **47.00% of it — 6.46% of
   ALL engine mispredicts — is one insertion-walk exit branch.**

**Landed (§7):** `GoodQuiet` and `BadQuiet` stop scanning each other's half of
the quiet region. Node-identical, eval-hash-identical, `Bench: 587683`
bit-for-bit.

Measured on two independent trees (before and after study 32's merge), 15 and
16 paired rounds:

| counter | arm vs base | null vs base | spread |
|---|---:|---:|---|
| `instructions:u` / node | **−0.267% / −0.258%** | −0.083% / −0.081% | **zero** |
| `branches:u` / node | **−0.649% / −0.618%** | −0.013% / −0.010% | **zero** |
| `L1-dcache-loads:u` / node | −0.135% / −0.013% | −0.129% / +0.016% | small |
| `cycles:u` / node | +1.058% / −0.546% | +0.802% / +1.753% | **±15 pts on the NULL** |

**No cycles claim is made, and §6 is why.** The two panels put the arm's cycles
median on *opposite sides of zero* for identical code, and in both the NULL —
a binary computing the identical tree — moved as far or further than the arm.
Per method rule 1 the instrument cannot answer a sub-1% question on this box
tonight, so the arm is carried by two exact counters (zero spread across 31
rounds on two trees) plus a direct before/after census of the work removed.

**Not taken, with the number that priced it (§8):** the branchless/vector
rewrite of the insertion walk — the single biggest mispredict site in
movepick — is the right next arm and it is *unmeasurable tonight*, because it
trades instructions UP for mispredicts DOWN and only cycles can adjudicate
that. Building it would have produced a number I could not read.

---

## 1. Prior art, and the trap in it

`init_quiets` has been worked before and the brief was right that study 7 got
its mechanism wrong. The live comments in `movepick.rs` are the record:

* **study 20 arm 7** — the partial sort's bounds checks, respelled so LLVM
  discharges them.
* **study 23 arms 3, 4b, 7** — the P-ORD prior outlined out of the scoring
  loop, the `PtRow` fusion, and the rejected `score_quiets` split. Study 23 is
  the direct predecessor: **its code landed on `main` but its write-up did
  not** (it lives only on the unmerged branch `cca9abd`). Read it there before
  touching this file; it will save you two builds.
* **study 27 arms 2, 3, 5** — the `ExtMove` layout with `value` at offset 0 and
  a named pad, literal stage assignments instead of `bump()`, and
  `#[inline(always)]` on `next_move`.

Study 23 also left a ranked remainder, and this study closes three of its
items with measurements (2, 5, 6) and lands work on none of the two it
predicted would pay. That is worth saying plainly: the remainder was a good
list and the arithmetic in it was mostly right; what it lacked was the
consumption census, which changes what is worth doing.

---

## 2. Method

| script | what it does |
|---|---|
| `tools/perf34/build34.sh` | memory-capped build (`systemd-run` cgroup 6 G, `oomlast.sh`, `-j2`), shipped profile, snapshot to `bin34/<label>` |
| `tools/perf34/ident34.sh` | the identity gate: node count + wl50 eval hash |
| `tools/perf34/rec34.sh` | `perf record` one event into a persistent `perf.data` |
| `tools/perf34/regions34.py` | buckets a `perf annotate` dump by innermost enclosing back-edge |
| `tools/perf34/panel34.sh` | the paired within-round panel |
| `tools/perf34/med34.py` | per-round pairing + median reduction |

Two method points are load-bearing.

**The panel is paired INSIDE the round.** Study 29 nearly shipped a −1.67%
cycles result that was position-in-round. Rotating the order across rounds is
not enough on its own: rotation still ends up comparing label A's round-`r`
number against label B's round-`r′` number when you median each label
separately. `med34.py` differences every label against the reference label's
run **from the same round** and medians the per-round deltas, so drift that
walks across the session cancels instead of landing on whichever arm occupied
that slot.

**The null perturbs the same machinery the arm does.** Study 23 §2.2: "a null
that does not perturb the same machinery the arm perturbs is not a null". A
byte-different build made by reordering two `let` bindings would not have moved
the stage machine at all. So the null here is a cargo feature, `null34`, that
builds the arm's struct layout, its two extra fields and its rewritten
`GoodQuiet` loop **with the optimisation switched off** — same code shape, old
amount of work.

The census itself is an instrument and is gated the same way: `census34` adds
relaxed atomics on paths that make no search decision, and `ident34.sh`
confirms the instrumented binary reproduces the plain build's node count and
eval hash exactly. If a counting build moved the tree it would be measuring
something else.

---

## 3. The census — how much of `init_quiets` is ever used

Study 23 §3 measured how much work `init_quiets` **does**. It never measured
how much of that work the search ever **looks at**. Per `init_quiets` call, on
the pre-rebase frame:

| quantity | per call | per node |
|---|---:|---:|
| `init_quiets` calls | — | 0.1740 |
| quiets generated and scored | **22.57** | 3.927 |
| placed in the sorted prefix | **9.31** | 1.620 |
| inspected by `GoodQuiet` | 10.16 | 1.768 |
| emitted by `GoodQuiet` | 4.22 | 0.734 |
| inspected by `BadQuiet` | 5.43 | 0.945 |
| emitted by `BadQuiet` | 3.19 | 0.555 |
| **ever inspected** | **15.59 (69.1%)** | |
| **ever emitted** | **7.41 (32.9%)** | |

and the shape of it, cumulatively — `HIST_GQ`, the number of elements a picker
inspected in `GoodQuiet` before the node ended:

| inspected ≤ | 1 | 2 | 4 | 6 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| share of calls | 20.5% | 34.0% | 46.5% | 52.8% | 58.4% | 75.1% | 88.0% |

**One in five `init_quiets` calls generates ~22 quiets, scores all of them,
partial-sorts nine, and consumes exactly one.** Half consume four or fewer.

Two derived facts that decided what to build:

* `skip_quiet_moves()` is called on **50.96%** of pickers (LMP), and only
  38.13% walk the whole quiet region.
* `init_captures` runs **twice as often** as `init_quiets` (0.3510 vs 0.1740
  per node) and scores **1.52** captures per call. Study 23 called it "almost
  entirely fixed overhead"; the fixed overhead is `generate_captures`, and
  `MoveList::new` is *not* part of it — study 20 arm 4 already removed the
  512-byte zero store, verified still absent.

### 3.1 The waste is real, and it should NOT be removed

Two thirds of the scoring loop's output is never read, and the scoring loop is
3.25% of engine cycles (§4), so ~2.2% of engine cycles is spent scoring quiets
nobody looks at. That is the largest single number in this study and it is a
trap.

To emit the *best* quiet first you must score every quiet — there is no
ordering-preserving way to score lazily. Any lazy scheme buys cycles by
degrading move ordering, and ordering is currently at **86.44% first-move
cutoffs**. Against ARCHON's own calibration constant (~120 Elo per doubling of
speed, `exp-0154`), 2.2% of cycles is worth ≈ 3.8 Elo. Degrading ordering
quality to collect it would cost far more than that. **The waste is structural
and buying it back is Elo-negative**; this is recorded so the next study does
not spend a night on it.

---

## 4. What `init_quiets` is made of, on the shipped build

`init_quiets` is **6.59% of engine cycles** — the sixth largest symbol, and
the largest outside the NNUE and `search` itself. (The brief quoted 6.35%;
confirmed.) It inlines `generate_quiets`, the offense reverse-fan, the
per-quiet scoring loop **and** `partial_insertion_sort`, so the symbol share
alone says nothing about mechanism.

Splitting the annotated `cycles:u` profile by address — no build, no code
change — gives:

| part | share of `init_quiets` | share of engine cycles |
|---|---:|---:|
| `generate_quiets` + setup | 19.64% | 1.29% |
| per-quiet scoring loop | 49.34% | **3.25%** |
| `partial_insertion_sort` | 30.30% | **2.00%** |
| out-of-line tail | 0.69% | 0.05% |

### 4.1 This corrects study 23

Study 23 §5 measured the sort at **2.84% of engine cycles** by marking it
`#[inline(never)]` to give it a symbol, and said in its own caveat that doing
so defeats the `limit = i32::MIN` specialisation the capture and evasion
callers get (with the constant inlined, `tmp.value < i32::MIN` folds to
`false` and the filter branch vanishes), so the shipped figure "is therefore
somewhat below 2.84%". It is: **2.00%**, measured on the shipped build with no
code change at all. The ranking in that section survives; the magnitude was
inflated by 42%.

The address split is what makes this possible and it is reusable —
`regions34.py` buckets an annotation by innermost enclosing back-edge, which
is how the four mechanisms inside one symbol were told apart without the
`#[inline(never)]` distortion that cost study 23 +23.5 i/n.

### 4.2 A note on the scoring loop, since it is the biggest piece

The disassembly shows **~11 stack reloads of loop-invariant plane base
pointers per scored quiet**, on top of the 9 semantically necessary loads (2
mailbox + 7 history). This is study 23 arm 3's spill traffic, still present and
larger than that study reported. It is *not* removable: the loop needs seven
plane pointers plus the buffer, list, board, index and bound live at once, and
x86-64 has 15 usable GPRs. The five continuation-history planes are rows of one
array, so they could share a base — but each address computation still needs
its own offset in a register, which is the same number of loads. Priced and
dropped without a build.

---

## 5. The mispredict localisation

`perf record -e ex_ret_brn_misp:u`, same frame:

| symbol | share of all engine branch mispredicts |
|---|---:|
| `search::<NonPv>` | 17.62% |
| **`init_quiets`** | **13.75%** |
| `ThreatContext::at_full` | 10.23% |
| `do_move` | 7.14% |

Inside `init_quiets`:

| part | share of `init_quiets` mispredicts | share of ALL engine mispredicts |
|---|---:|---:|
| `generate_quiets` + setup | 17.36% | 2.39% |
| scoring loop | 18.87% | 2.59% |
| **`partial_insertion_sort`** | **61.52%** | **8.46%** |

and inside the sort:

| sub-region | cycles (of `init_quiets`) | mispredicts (of `init_quiets`) |
|---|---:|---:|
| the `value < limit` filter loop | 6.16% | 8.68% |
| insertion prologue (hole-punch) | 4.74% | 5.84% |
| **the shift walk** | **11.25%** | **27.45%** |
| **its exit + the final stores** | **8.15%** | **19.55%** |

**The shift walk and its exit are 19.40% of `init_quiets`' cycles — 1.28% of
all engine cycles — and 47.00% of its mispredicts, which is 6.46% of ALL
engine branch mispredicts.** That is 1.28% of cycles for ~2 instructions per
shift step: it is not doing work, it is mispredicting, exactly as study 23
diagnosed but now localised to a single branch (`head[q-1].value < tmp.value`)
rather than to the function.

---

## 6. The instrument finding: cycles could not answer this tonight

The brief says to gate on cycles, median of ≥ 6 rounds. That is the right
default and it is what the panel ran. It did not survive the null.

Paired within-round, order rotated. **Pre-rebase**, 15 rounds, vs `base`:

```
label   n            i/n              cyc/n             br/n           loads/n
null   15  -0.083%[-0.08,-0.08]  +0.802%[-9.24,+19.52]  -0.013%   -0.129%[-0.35,+0.18]
arm1   15  -0.267%[-0.27,-0.27]  +1.058%[-4.71,+23.08]  -0.649%   -0.135%[-0.44,+0.12]
```

**Post-rebase**, 16 rounds, vs `mbase` (pure `main`):

```
label   n            i/n              cyc/n             br/n           loads/n
mnull  16  -0.081%[-0.08,-0.08]  +1.753%[-7.31,+11.12]  -0.010%   +0.016%[-0.45,+0.25]
marm   16  -0.258%[-0.26,-0.26]  -0.546%[-6.57,+10.83]  -0.618%   -0.013%[-0.45,+0.22]
```

Read the **null** row as the error bar. It is a binary that computes the
identical tree — same nodes, same eval hash — and its cycles median came out
**+0.802% over a 28.8-point range** on one tree and **+1.753% over an 18.4-point
range** on the other.

Then look at what the arm's cycles column did across those two panels. **Pre-rebase
it reads +1.058% and post-rebase −0.546%: for identical source, the cycles
instrument put the same change on opposite sides of zero.** Net of the null it
would read +0.256% on one panel and −3.060% on the other. That is as clean a
demonstration as this study could ask for that cycles is not measuring the arm
here, and it is exactly the shape of the error study 22 made and retracted and
study 29 caught at n=13. **I am not quoting the −3.060%.** It is the more
flattering number and it is the one the null says I have not earned.

`ex_ret_brn_misp:u` is contaminated the same way: the null moved +0.121% and
+0.534% on the two panels, over ranges of 14.2 and 9.4 points. So the mispredict
*counter* cannot adjudicate an arm tonight even though the mispredict *profile*
(§5 — a within-run attribution, immune to between-run drift) is perfectly
usable. That distinction is what let §5 exist at all.

What survives is exact: **`instructions:u` and `branches:u` had ZERO spread
across all 31 rounds on both trees** — every round's delta identical to three
decimal places — and the two trees agree with each other to within 0.03
percentage points on an arm that was never rebuilt for the second panel's
purpose. Those, plus a direct before/after count of the work removed (§7), are
what the landed arm is carried on.

This is not a complaint about the box; it is a scheduling fact. Four perf
studies sharing eight cores cannot all have a cycles instrument, and the honest
move is to say which claims that permits and which it does not.

---

## 7. Landed — `GoodQuiet` and `BadQuiet` stop scanning each other's half

`GoodQuiet` emits quiets with `value > GOOD_QUIET_THRESHOLD` and `BadQuiet`
emits the exact complement. Between them they inspected **every quiet twice**:

* `GoodQuiet` walked to the **end of the region** rejecting every remaining
  element, and then the caller reset `cur` — throwing the position away.
* `BadQuiet` restarted at `end_captures` and re-rejected every above-threshold
  element `GoodQuiet` had just emitted.

Both halves are dead whenever the sort's `limit` is at or below the threshold.
The sort leaves a descending prefix of everything scoring `>= limit` and a tail
entirely **below** `limit`; so when `limit <= GOOD_QUIET_THRESHOLD` no tail
element can pass `GoodQuiet`, and within the prefix descending order makes the
sub-threshold elements a contiguous suffix. The good set is then exactly one
prefix of the region and the bad set exactly the rest, split at a single index
— which `GoodQuiet` discovers for free by breaking on its first sub-threshold
element, and which `BadQuiet` resumes from.

`limit = QUIET_SORT_DEPTH_FACTOR · depth` is `−3560·depth`, the threshold is
`−14000`, so the condition holds from **depth 4** up. It is computed, not
assumed: both are SPSA-tunable under `tune`, and at depths 1–3 the flag is
false and both stages take the old full scan. That is what keeps shallow nodes
bit-identical rather than merely equivalent.

**Mechanism, measured directly** (census build, before vs after, per
`init_quiets` call):

| | before | after |
|---|---:|---:|
| `GoodQuiet` elements inspected | 10.16 | **6.36** |
| `BadQuiet` elements inspected | 5.43 | **3.56** |
| `GoodQuiet` moves emitted | 4.22 | 4.22 |
| `BadQuiet` moves emitted | 3.19 | 3.19 |
| inspected / scored | 69.08% | **43.93%** |

**5.67 inspections per call — 0.99 per node — were pure waste.** I predicted
7.4 before building it and over-priced it by 30%; the table is measured.

**Result** (§6 for why no cycles number), replicated on two trees:

| | pre-rebase (15 rounds) | post-rebase (16 rounds) |
|---|---:|---:|
| `instructions:u` / node vs base | **−0.267%** | **−0.258%** |
| `branches:u` / node vs base | **−0.649%** | **−0.618%** |
| `instructions:u` / node vs null | −0.184% | −0.178% |
| `branches:u` / node vs null | −0.636% | −0.608% |

all with zero spread across every round. The branch figure is the mechanism
check: 0.99 removed inspections per node × 3 branches each (loop test, TT-move
dedup, value test) predicts ≈ 3.0 branches/node against 3.64 measured
(0.618% of 627.68), the remainder being the removed loop-exit machinery.

Node-identical and eval-hash-identical on both trees, and `Bench: 587683`
reproduces bit-for-bit on the default build.

---

## 8. Priced, and not taken

Each of these was costed against a measurement, not waved away.

| candidate | the number that decided it |
|---|---|
| **Branchless / vector insertion walk** — the biggest mispredict site in movepick (§5) | **The right next arm, and unmeasurable tonight.** Every formulation trades instructions UP (a fixed-width branchless window is ~59 ops per insertion against ~11 for the current walk) for mispredicts DOWN. Only cycles can adjudicate that trade, and §6 shows cycles carrying a ±15-point null. It also covers only ~68% of insertions at width 8 (derived from `HIST_PFX`: mean prefix 9.31, so `E[min(P−1,8)] / E[P−1]` = 5.62/8.31). Building it would have produced a number I could not read. |
| **Lazy quiet scoring** — 67% of scoring output is unused | Structural (§3.1). ~2.2% of cycles ≈ 3.8 Elo at ARCHON's own 120-Elo-per-doubling constant; any lazy scheme pays for it in ordering quality, which is at 86.44% first-move cutoffs. Elo-negative. |
| **Interleaving the six `[pci][to]` history planes into one row** | 6 planes × 64 squares presummed per piece type is ~50 SIMD ops of setup × up to 6 types against 22.57 quiets × ~16 saved ops — roughly break-even at the measured mean list length, and a **loss** on short lists. Dropped without a build. |
| **Reusing the evasion list `pseudo_legal` already generated** | Real, and priced: **157,016 of the engine's 349,337 evasion generations (45%) are duplicates** (§9). Worth ~0.2% of engine cycles. Breaks the module's "a TT-move cutoff performs zero generation" contract unless the list is carried in the picker's own buffer — a delicate refactor that needs a cycles instrument to justify. |
| **Removing the `pseudo_legal(tt_move)` guard** — it rejects 8 of 714,496 | **Rejected on reading the code, not on the rate.** It fires, so TT collisions reach it; `Board::legal` only tests the pin ray for a non-king move and would NOT catch a move that fails to address an existing check, so a garbage TT move would be played. Load-bearing at 1 in 89k. |
| **Short-circuiting on non-Normal TT moves** (full `NON_EVASIONS` generation) | 7,065 calls, 0.0014/node. I expected castling TT moves to make this common; measured, it is negligible. Arm not built. |
| **`MoveList::new` zero store** | Already gone — study 20 arm 4. Verified present-tense rather than assumed. |

---

## 9. `MovePicker::new` — study 23 remainder item 6, closed

| quantity | count | per node |
|---|---:|---:|
| `MovePicker::new` | 2,322,840 | 0.4644 |
| … with a candidate TT move | 714,496 | 0.1428 |
| … TT move survived `pseudo_legal` | 714,488 | — |
| … in check + Normal TT move (⇒ a full evasion generation) | 157,016 | 0.0314 |
| … non-Normal TT move (⇒ a full `NON_EVASIONS` generation) | 7,065 | 0.0014 |
| `init_evasions` calls | 349,337 | 0.0698 |

The headline is the last two rows together: **45% of every evasion generation
the engine performs is a duplicate of one `pseudo_legal` performed for the same
position microseconds earlier** — generated, scanned once with `contains`, and
dropped.

---

## 10. Ranked remainder

1. **The insertion walk, branchless.** 1.28% of engine cycles and 6.46% of all
   engine mispredicts, in one branch. Needs a quiet box, not a new idea. Build
   the fixed-width-8 window (`se <= 8 && n > 8`, ~68% insertion coverage, the
   self-masking select form is written out in §8) and adjudicate on cycles.
2. **The duplicate evasion generation.** ~0.2% of cycles, mechanism fully
   measured (§9). Carry the `pseudo_legal` list into the picker's buffer.
3. **`QUIET_SORT_DEPTH_FACTOR` is mistuned for cost.** The sorted prefix is
   9.31 elements and `GoodQuiet` consumes 4.22 of them; the remainder is sorted
   solely for `BadQuiet`, whose moves are searched last at maximum reduction.
   Aligning the sort limit with `GOOD_QUIET_THRESHOLD` would roughly halve the
   sort. This is a **tunable**, so it is a functional change and games arbitrate
   — but nobody has known its cost before, and now it is 2.00% of engine
   cycles.
4. **The stage dispatch**, study 23 remainder item 4, still open: `next_move`
   re-enters the 16-arm match through a jump table on every call (2.335 per
   node) even though study 27 arm 3 made every fall-through a literal.
5. **`init_captures` at 0.94% of cycles for 1.52 scored captures** — 125 cycles
   per call of fixed overhead, essentially all `generate_captures`. The lever
   there is movegen, not movepick.

---

## 11. Gates

`CHECK_NNUE_GATE=1 bash tools/check.sh` on the rebased tree (post study 32),
**exit 0, "All checks passed"**:

| gate | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| feature-matrix `cargo check` | clean |
| `cargo test` (debug: unit + fast perft tier) | pass |
| release perft suite (`--ignored perft_full_suite`) | pass |
| `cargo test --features sota,arch-d48x` | pass |
| bench perft fingerprint | 1839507 |
| `bench 16 1 6 default depth` | **587683** — the pin, bit-for-bit |
| wl50 eval hash | `eb981b119bd66d6eade8e314aa32d254e7fc4821c29e33ac20b5ccc4c84f4f37` — the pin |
| d13 node total | **1985398** — the pin |
| node identity, study frame | 5,001,624 on `mbase`, `mnull` and `marm` alike |
| pre-rebase identity | 5,001,562 / `5ca978e012da1bb0` on base, null, arm and every census build |

The eval-hash and d13 rows are the tripwire the brief asked for: a movepick
change must not move them, and it did not.

---

## 12. Reproducing

```bash
bash tools/perf34/build34.sh base                                  # shipped
bash tools/perf34/build34.sh null   "null34,sota,arch-d48x"        # the null arm
bash tools/perf34/build34.sh census "census34,sota,arch-d48x"      # the instrument
bash tools/perf34/ident34.sh base null census                      # identity gate

./bin34/census bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn 2>&1 >/dev/null | grep '^C34'       # §3, §9

bash tools/perf34/rec34.sh base cycles:u                           # §4
bash tools/perf34/rec34.sh base ex_ret_brn_misp:u 100000 2003      # §5
python3 tools/perf34/regions34.py <annot.txt>                      # §4 split

bash tools/perf34/panel34.sh 15 base null arm1                     # §6, §7
python3 tools/perf34/med34.py docs/research/perf-artifacts-34/panel.txt base
```

Raw output: `docs/research/perf-artifacts-34/` — `census.txt` and
`census-post-arm1.txt` (the before/after of §7), `panel.txt` and
`panel-post-rebase.txt` (every panel row, in order),
`annot-init_quiets-cycles.txt` and `annot-init_quiets-misp.txt` (the two
annotations §4 and §5 are computed from).

The `perf.data` files are gitignored (15 MB); regenerate with `rec34.sh`.
`bin34/` and `target34-*/` are build outputs and are not tracked.
