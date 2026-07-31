# Perf study 55 — how far the two mechanisms generalise: one has a size limit, one is everywhere

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `e8ab822b` (study 54 merged,
plus the owner's cycles-noise measurement) · **Net/arch:** D48X @
`network_scale=600`, `sota,arch-d48x`, ACC 1280 · **Instruments:**
`tools/perf55/` (rebased from study 54) · **Artifacts:**
`docs/research/perf-artifacts-55/` · **Branch:** `perf-study-55-generalise`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, pinned to
physical core 2. Non-PGO release, fat LTO, CGU = 1, `-C target-cpu=native` — the
build profile studies 52, 53 and 54 adjudicated on.

**Frame cross-check (rule 8).** `base55` measures **6030.49 instructions/node**
and its `.text` md5 is **`1ce4526f`** — exactly the md5 study 54's gate log
records for `final54`, the binary it landed. Study 54 landed −46.73 i/n against
its own `base54` of 6077.24; 6077.24 − 46.73 = **6030.51**. The frame reproduces
to two decimal places across a session boundary and through a landed change.

---

## ★ 0. THE ANSWER, FIRST

### 0.1 ★ Mechanism A generalises — but it is not "call sites", it is BODY SIZE

Study 54 measured −20.8 i/n from making `make_move_with::<H>` reachable from a
single call site, which let LLVM's last-call-to-static bonus inline all 2466
bytes of it into `do_move`. The commission asked how far that generalises.

**It generalises to small bodies and INVERTS on large ones.** Seven arms, every
one node-identical, every one measured against the same base in a blocked Latin
panel with a byte-identical null riding along:

| arm | what it forces inline | body sizes | sites | **i/n** | verdict |
|---|---|---|---|---:|---|
| `inlas55` | `Stage::bump`, `MovePicker::select`, `Board::legal`, `is_shuffling` | 68–267 B | 2 each | **−19.3** | **WIN** |
| `inlab55` | `stat_score`, `upcoming_repetition`, `new_probcut` | 268–802 B | 2 each | **−25.3** | win on instr, **+41 loads/node** |
| `inlb55` | `unmake_move`, `tt::probe`, `TtWriter::write`, `capture_stage`, `gives_check` | 87–422 B | 3–10 | **−12.6** | **WIN** |
| `inle55` | `see_ge` | 941 B | 22 | **+4.5** | loss |
| `inld55` | `check_time`, `correction_value`, `attackers_to_exist`, `pseudo_legal` | 341–815 B | 4–10 | **+25.0** | loss |
| `inlc55` | `do_move` itself | 3516 B | 8 | **+25.7** | loss |
| `inlall55` | `inla55 + inlb55 + inlc55` | — | — | **+5.7** | loss (dominated by `inlc55`) |

The turn is sharp and it is not at the call-site count — `inlb55` wins at 3–10
sites while `inlab55` costs loads at 2. **What separates the winners from the
losers is how much state the inlined body adds to the already-huge `search::<NT>`
frame.** `inlc55` inlines 3516 bytes of `do_move` (which already contains study
54's 2466 bytes of `make_move_with`) into a 16 KB body and pays **+25.7 i/n** in
spills for the call it deleted.

**Three arms in this study cost more than study 54's single application gained.**
Anyone reading study 54 as "inline more things" would have shipped a regression.

### 0.2 ★ "One call site ⇒ LLVM inlines it" is FALSE, and the tree already disproves it twice

The naive reading of study 54's mechanism does not survive contact:

* **`make_move_with::<TtPrefetch>` now has exactly ONE call site** — the ProbCut
  verification probe, after study 54 moved `do_move` onto `TtPrefetchDirty` —
  **and it is still out of line at 2466 bytes** in the shipped binary.
* **`MovePicker::select` has five monomorphizations with one caller each**
  (Rust closure types are unique per source expression, so the three textually
  identical `|_, _| true` closures are three distinct types). All five were out
  of line.

In both cases the call site is cold or cheap and LLVM declined. **The enabling
condition in study 54 was a single *hot* call site, not a single call site.**

### 0.3 ★ The mechanism was ALREADY running, unnoticed, at the largest scale in the engine

`NodeType` has **three** impls, not two — `Root`, `Pv`, `NonPv` — and
`search::<Root>` *is* instantiated, at `src/search/mod.rs:2220`. Yet **there is
no `search::<Root>` symbol in the binary.** It has exactly one call site, so the
last-call-to-static bonus inlined the whole ~16 KB body into
`iterative_deepening`, which is **29,459 bytes** against `search::<NonPv>`'s
16,264.

Nobody put that there deliberately. It is study 54's mechanism operating on the
biggest function in the engine, and it is the reason `do_move`, `init_captures`
and `init_quiets` all show *extra* machine call sites charged to
`iterative_deepening` in a call-graph census — the root search's copies of them.

### 0.4 The structural cause of mechanism A's surface

`search::<NT>` is two out-of-line 16 KB monomorphizations of one body (plus the
Root copy folded into `iterative_deepening`). **Every helper called from the
search body therefore has at least two machine call sites and can never qualify
for the last-call-to-static bonus** — including `stat_score`, `is_shuffling` and
`check_time`, which already carry `#[inline]` and which LLVM declined anyway.
That is the population mechanism A acts on, and it is why `inlas55` buys 19.3
instructions/node for **+184 bytes of code**.

---

## 1. THE INSTRUMENT, AND ITS NULL

**Published null (n=16, `panel-null.txt`), `null55` a byte-for-byte `cp` of
`base55`, riding in the rotation:**

| counter | null | reading |
|---|---:|---|
| **`instructions:u`** | **−0.000 %, se 0.000** | **exact** |
| `ls_dispatch.ld_dispatch:u` | +0.044 %, se 0.090, 8/16 | clean |
| `ls_dispatch.store_dispatch:u` | +0.089 %, se 0.088, 8/16 | clean |
| `L1-dcache-loads:u` | +0.159 %, se 0.039 | **a real floor** |
| `cycles:u` | +0.558 %, se 0.703 | cannot resolve anything |

Better than a percentage: across those 16 paired rounds the byte-identical null
moved `instructions:u` by **at most 1250 counts out of 3.0 × 10¹⁰** — 0.000004 %,
or **±0.00025 instructions/node.** Every effect claimed in this study is between
4,500× and 100,000× that. `instructions:u` is not merely low-noise here, it is
very nearly a deterministic function of the binary.

**No cycles claim is made anywhere in this study**, for the reason the
commission gave and this study's own null reproduces. Ordering is a **blocked
Latin square** (rotate within a block, reshuffle the base between blocks);
deltas are paired **within** round.

### 1.1 The inert control IS inert, twice over

`ctl55` is this study's edited tree built with **no arm feature**, against
`base55` built from the untouched tree.

* **`ctl55`'s `.text` md5 is `1ce4526f` — byte-identical to `base55`'s.**
* It measures **−0.000 %, se 0.000** on `instructions:u` in `panel-r1.txt`.

So the whole `#[cfg_attr]` scaffolding is provably free, and **every arm in this
study is a straight arm-minus-base contrast with no decomposition needed.**
(Study 53's control failed this test at −6.9 i/n on its own; study 54's passed.)

---

## 2. MECHANISM A, ARM BY ARM

*(Panels: `panel-r1.txt` n=16 for `ctl55`/`inla55`/`inlb55`; `panel-r2.txt` n=6
for `inlc55`/`inld55`/`inle55`/`inlall55`; `panel-r3.txt` for the
`inlas55`/`inlab55` split. `instructions:u` carries se 0.000 in all three.)*

| arm | instr % | **i/n** | neg/n | ld_dispatch % | L1d % | store % |
|---|---:|---:|---:|---:|---:|---:|
| `ctl55` | −0.000 | 0.0 | 11/16 | +0.039 | +0.020 | +0.049 |
| **`inla55`** | **−0.310** | **−18.7** | **16/16** | +0.939 | +0.321 | +0.615 |
| **`inlb55`** | **−0.209** | **−12.6** | **16/16** | −0.132 | −0.132 | −0.255 |
| `inlas55` | −0.320 | −19.3 | 4/4 | +0.075 | **−0.297** | +1.046 |
| `inlab55` | −0.420 | −25.3 | 4/4 | **+1.816** | +0.720 | −0.091 |
| `inlc55` | +0.427 | +25.7 | 0/6 | +1.069 | — | — |
| `inld55` | +0.415 | +25.0 | 0/6 | +1.088 | — | — |
| `inle55` | +0.075 | +4.5 | 0/6 | +1.151 | — | — |
| `inlall55` | +0.095 | +5.7 | 0/6 | +0.940 | — | — |

### 2.1 The load cost has an owner, and it is the BIG bodies

`inla55` deletes 18.7 instructions/node and **adds ~21 loads and ~5.5 stores per
node** — reported here because it is the one number that argues against the arm.
The split says where it comes from: **`inlab55` alone is +1.816 % ld_dispatch
(+41 loads/node)** while **`inlas55` is +0.075 %, indistinguishable from the
null, and −0.297 % on L1d.** The three bodies in `inlab55` (`stat_score` 505 B,
`upcoming_repetition` 802 B, `new_probcut` 268 B) are the ones fat enough to
spill when folded into `search::<NT>`.

### 2.2 The two halves are strongly NON-additive, and the combination is worse than either

| | instr % | i/n |
|---|---:|---:|
| `inlas55` alone | −0.320 | −19.3 |
| `inlab55` alone | −0.420 | −25.3 |
| **sum** | **−0.740** | **−44.6** |
| **`inla55` measured (both features on)** | **−0.310** | **−18.7** |

**The combination gives back 0.43 percentage points — 26 instructions/node —
against the sum of its parts.** Study 54 found instructions adding "to the third
decimal place" across *its* two arms and this study reproduces that for
`inlas55 + inlb55` (§4), so additivity is not a property of the counter: it is a
property of whether the arms compete for the same resource. These two both
inline into the same `search::<NT>` frame, so they do.

This is why every landing figure in §4 is **built and measured**, never summed
(study 53 §2.2's rule).

---

## 3. MECHANISM B — the spelling defect, and it is BIGGER than mechanism A

Study 54 measured a 30.7 i/n swing between two spellings of one algorithm,
because the losing shape indexed `list[i]` twice per move with bounds checks
where `list.iter()` has none. A census applied that lens to every indexed access
in the hot path. Four arms, `panel-r4.txt`, n=4, `instructions:u` se 0.000:

| arm | site | instr % | **i/n** | neg/n |
|---|---|---:|---:|---:|
| **`bcbuf55`** | `MovePicker::slot`/`set_slot`/`swap_slots` masked | **−1.403** | **−84.6** | 4/4 |
| **`bcmisc55`** | `reductions[]`, `LMR_DIVISOR[]`, `DeltaBuf` write | **−0.742** | **−44.7** | 4/4 |
| **`bcsort55`** | `partial_insertion_sort`'s double `head[q-1]` read | **−0.574** | **−34.6** | 4/4 |
| `bccont55` | `update_continuation_histories` zip + flat `piece_to` | +0.096 | +5.8 | 0/4 |
| `bcall55` | all four together | +0.012 | +0.7 | 0/4 |

**`bcbuf55` alone is −84.6 i/n — larger than study 54's entire landed total
(−46.73), from adding `& BUF_MASK` to three accessors.** `MAX_MOVES` is 256 and
the region invariant on `buf` already proves every access is in range, but the
index reaches the accessor as a struct field (`self.cur`) or `base + i`, so LLVM
could prove nothing and every one of the ~50–80 buffer touches per node carried
a `cmp`/`jae` and a panic edge. This is the same fix `attacks.rs` carries as
`TABLE_MASK` (study 53's `magic53`) and `history.rs` as `CONT_MASK` — it was
simply missing on the funnel every move in every stage of every node passes
through. The binary gets **3,576 bytes SMALLER**.

`bcsort55` deserves its own note. `partial_insertion_sort`'s shift loop reads
`head[q - 1]` twice per shifted element — once in the loop condition, once in the
body — twenty lines below a comment in the same function reading *"One read of
`moves[p]`, not two"*. Reading it once is **−34.6 i/n**. This is **not** either
rewrite study 23 §6 measured and rejected (`copy_within` +91 i/n,
`partition_point` +77.8 i/n): the algorithm, the comparison order and the
resulting permutation — including the `BadQuiet` tail, which is observable — are
all untouched. Only the load count changes.

### 3.1 The one that lost, and why it is still a finding

`bccont55` is **+5.8 i/n**, a small loss, and it is the arm that looked most like
study 54's on paper: `update_continuation_histories` indexes `stack[cur - i]`
twice per iteration, and `stack` is a `&[Stack]` — a *runtime-length* slice, so
strictly worse than the fixed-size `MoveList` study 54 fixed. Converting it to
one subslice plus a reversed `zip` did not pay. **The shape being present is not
sufficient; the loop trip count (6) and the surrounding register pressure decide
it.** Reported because it is the arm that most directly tests the mechanism's
naive form, and it failed.

### 3.2 ★ The arms are violently NON-additive, in both directions

This is the study's most important methodological result and it goes both ways:

| combination | predicted from parts | **measured** | error |
|---|---:|---:|---:|
| `bcall55` = all four B arms | −2.62 % | **+0.012 %** | **+2.63 pp** |
| `bc3_55` = B minus `bccont55` | −2.72 % | **−0.925 %** | +1.79 pp |
| `landa55` = `inlas55` + `inlb55` | −0.529 % | **−1.264 %** | **−0.735 pp** |
| `inla55` = `inlas55` + `inlab55` | −0.740 % | −0.310 % | +0.43 pp |

**Study 54 reported instructions adding "to the third decimal place" across its
two arms. That does not generalise at all.** Four of this study's combinations
miss the sum of their parts by 0.4 to 2.6 percentage points — 26 to 159
instructions per node — and `landa55` misses it in the *favourable* direction by
44 i/n. Every one of these arms changes what LLVM inlines and how it allocates
registers inside two 16 KB search bodies, so they compete for, and occasionally
free up, the same resource.

**The operational rule this forces: on this codebase you may not sum arms, in
either direction. Build the combination you intend to ship and measure that.**
Study 53 §2.2 stated the rule; this study is the first to show it failing by
more than two percentage points, and the first to show it failing *pessimistically*.

---

## 4. WHAT LANDED — −83.8 instructions/node, the largest of the series

Three groups, promoted to unconditional, all node-identical:

* **`#[inline(always)]` on nine SMALL hot helpers.** From mechanism A's winning
  half: `Stage::bump`, `MovePicker::select`, `Board::legal`,
  `Worker::is_shuffling` (the pure `search::<Pv>`/`search::<NonPv>` duplication
  class) and `Board::unmake_move`, `Board::gives_check`, `Board::capture_stage`,
  `TranspositionTable::probe`, `TtWriter::write`.
* **`BUF_MASK` on `MovePicker::slot`/`set_slot`/`swap_slots`.** Mechanism B's
  largest arm: −84.6 i/n on its own, and the binary gets **3576 bytes smaller**.

### 4.1 The candidates, built and measured, never summed

`panel-r6.txt`, n=6, five labels, `instructions:u` se **0.000** on every arm,
6/6, p = 0.0312 each; byte-identical null at −0.000 %:

| candidate | contents | instr % | **i/n** |
|---|---|---:|---:|
| **`landbuf55`** ← **LANDED** | small-inline + `BUF_MASK` | **−1.390** | **−83.8** |
| `landa55` | small-inline only | −1.264 | −76.2 |
| `landmisc55` | small-inline + `bcmisc55` | −1.032 | −62.2 |
| `final55` | small-inline + all three B winners | −1.018 | −61.4 |

**The judgement call, stated as one.** `landbuf55` is best on `instructions:u`,
the counter study 52 §4 licenses, and **no memory counter regresses** with it.
But `landa55` is better on all three memory counters (L1d −1.242 %, ld −1.375 %,
st −1.167 %, every one ~1.2–1.4 % and coherent) and worse on the gate counter by
0.126 pp = 7.6 i/n. I landed `landbuf55` because instructions is the counter with
a zero null on this box and `landbuf55` does not lose anywhere; **a reader who
weights the memory counters above 7.6 i/n should land `landa55`, and the numbers
to do that are in this table.**

### 4.2 "I measured what I landed"

Promotion changes the binary — removing `#[cfg]` scaffolding relocates panic
`Location` constants (study 35 §5.1), so `landbuf55`'s `.text` md5 is `1d5e7b18`
and the promoted tree's is `4e4aad52`. `panel-final.txt` runs them against each
other in one rotation:

| `panel-final.txt`, n = 4 | instr % | i/n | L1d % | ld % | st % |
|---|---:|---:|---:|---:|---:|
| `null55` (byte-identical) | −0.000 | 0.0 | +0.161 | +0.423 | +0.460 |
| `landbuf55` (the arm) | **−1.390** | −83.8 | −0.691 | −1.131 | −0.323 |
| **`promoted55` (the tree)** | **−1.390** | **−83.8** | −0.888 | −1.884 | −1.087 |

**Both read −1.390 %, se 0.000, in the same panel.** Every memory counter moves
the right way net of this panel's own (positive) null.

`cargo fmt` ran *after* `promoted55` was built, so the binary was rebuilt and
compared: `.text` md5 `4e4aad52` unchanged. The formatting was cosmetic.

---

## 5. GATES

| gate | result |
|---|---|
| `cargo fmt --check` | **PASSED** |
| `cargo clippy --all-targets -- -D warnings` | **PASSED** |
| feature-matrix `cargo check` | **PASSED** |
| `cargo test` (debug: unit + fast perft tier) | **PASSED** — 614 passed, 0 failed |
| **D24 perft ritual** (`board`/`movegen` touched) | **`perft_full_suite` PASSED** |
| `Bench:` trailer (depth-6 default build) | **587683** — **unmoved** |
| **wl50 eval hash** | **`2e1ede9521624dd8…93cf79c4`** — **unmoved** |
| **d13 node total** | **1 739 588** — **unmoved** |
| frame node identity, all 20 binaries | **5 001 562** exact |
| inert control `ctl55` `.text` md5 | **`1ce4526f` = `base55`** |
| byte-identical null, `instructions:u` | **≤1250 counts in 3.0 × 10¹⁰** |
| `promoted55` reproduces `landbuf55` | **−1.390 % both**, se 0.000 |

**No pin moved, and none should have** — every change here is
behaviour-preserving, so a moved pin would have been evidence that an
equivalence was wrong.

**One honest note on how the two NNUE pins were obtained.** `tools/check.sh`'s
final phase — the 727-test `--features sota,arch-d48x` suite — was still running
when this was written, at roughly 5 tests/minute, because the box is shared with
the live D49 trainer and `CHECK_TEST_THREADS=1` forces one thread. It had
reported **zero failures** up to that point. Rather than block, the two pins were
computed **directly from the promoted binary** with check.sh's own invocations:

```bash
./bin55/promoted55 nnue-eval engines/canonical/net.cenn \
    < tools/nnue/calib/wl50.fens | sha256sum          # 2e1ede95…93cf79c4
./bin55/promoted55 bench 16 1 13 tools/nnue/calib/wl50.fens depth \
    engines/canonical/net.cenn                         # 1 739 588
```

Both match. The sota unit-test suite is the one gate line not personally seen to
completion; it should be confirmed green before merge.

---

## 6. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **Why the combinations are non-additive by up to 2.6 percentage points.**
   §3.2 establishes *that* they are, in both directions, and rules out the
   counter (`instructions:u`'s null is ±0.00025 i/n). I did not decompose which
   register or which inlining decision the arms are competing for. A static
   stack-traffic census of `search::<NonPv>` across the pair — the instrument
   study 54 §0.4 built and got a null from — is the obvious next step.
2. **Why `bccont55` loses.** It has study 54's exact shape on a strictly worse
   substrate (a runtime-length slice) and it is +5.8 i/n. Trip count 6 and
   register pressure are the candidate explanations; neither is measured.
3. **`bcmisc55` (−44.7) and `bcsort55` (−34.6) are unlanded wins.** Each is
   large alone and each is blocked only by its interaction with `BUF_MASK`. They
   are retained as features. **Do not assume they stack** — that is the exact
   error §3.2 documents.
4. **No cycles claim, anywhere.** This study's null read +0.558 % (se 0.703) on
   `cycles:u`. The landed change is −1.390 % of instructions; at n = 16 paired
   rounds the instrument still could not resolve it, and I did not run the ~23
   rounds the owner's σ = 1.57 % measurement says would be needed.
5. **No PGO measurement.** Every number here is non-PGO — the third study in a
   row to owe this, and it now matters most, because the dominant mechanisms are
   inlining and bounds-check elision, exactly what PGO re-litigates.
6. **No games.** Every change is a node-identical refactor, so games are not owed
   for correctness — only for the speed→Elo conversion at the standing ~120 Elo
   per doubling.

---

## 7. WHAT THE NEXT STUDY SHOULD RUN

1. **`bcmisc55` and `bcsort55`, as a landing question, not a measurement one.**
   Both are large alone. Find the combination with `BUF_MASK` that keeps them —
   or establish that the interaction is real and unavoidable. `bcmisc55` also
   **closes a latent panic**: `move_number` reaches 256 in a 256-move position
   and `reductions` is `[i32; 256]`. That is a correctness reason to land it
   independent of speed.
2. **The PGO pair**, now overdue across studies 52, 53, 54 and 55.
3. **Do not extend `#[inline(always)]` upward without measuring.** §0.1 has the
   size curve and it inverts. `do_move` is +25.7 i/n.
4. **The census in §0.3 is reusable.** `tools/perf55/` has no census script, but
   the method is: `objdump -C -d`, count `call`/`jmp` targets per symbol, join
   against `nm -C --size-sort -S`. It found every candidate in this study and it
   found `search::<Root>`'s disappearance, which nobody was looking for.

---

## 8. REPRODUCTION

```bash
git checkout perf-study-55-generalise
bash tools/perf55/build55.sh base55                      # or any labelled arm
bash tools/perf55/build55.sh landbuf55 inlas55,inlb55,bcbuf55   # pre-promotion
bash tools/perf55/exact55.sh 16 base55 null55 landbuf55 promoted55
python3 tools/perf55/med55.py docs/research/perf-artifacts-55/panel-final.txt \
        base55 null55 landbuf55 promoted55
CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh
cargo test --release -- --ignored perft_full_suite
```

`med55.py` fixes study 54 §3.4's reporting artifact: the mean-position line now
averages over the **adjudicated** rounds, not every parsed round.

