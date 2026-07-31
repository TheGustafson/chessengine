# Perf study 54 — the fusion, re-derived: a win, for four-fifths the wrong reason

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `5f6e988c` (study 53
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf54/` (rebased verbatim from study 53) ·
**Artifacts:** `docs/research/perf-artifacts-54/` · **Branch:**
`perf-study-54-fusion`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, pinned to
physical core 2. Non-PGO release, fat LTO, CGU = 1, `-C target-cpu=native` —
the same build profile studies 52 and 53 adjudicated on.

**Frame cross-check (rule 8).** `base54` measures **6077.24 instructions/node**,
which is *exactly* study 53's landed `final53` figure, and `base54`'s `.text`
md5 is **`121e12b7`** — the same md5 study 53's gate log records. The frame
reproduces to the last printed digit across a session boundary.

---

## ★ 0. THE ANSWER, FIRST

### 0.0 ★ What LANDED — −46.73 instructions/node, node-identical

Two independent changes, promoted to unconditional:

* **Hook C fusion** (§1) — the `DirtyPiece` is built inside `make_move_with`
  from step 3's own actors (`pc`, `(captured, capsq)`, `us`) and handed to the
  search through Hook C, so neither `state.moved` nor `state.captured` is read
  back after the make.
* **`init_quiets` run-hoisting** (§2) — the quiet scoring loop reads the mailbox
  once per PIECE-TYPE RUN instead of once per quiet, carrying `row` and `pci`
  across iterations.

Measured on the promoted tree, `final54` vs `base54`, same frame
(`panel-final.txt`, n = 20, byte-identical null riding in the rotation, null
**clean on every counter** in this panel):

| `final54` vs `base54` | median | se | neg/n | p | per node |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | **−0.769 %** | **0.000** | **20/20** | 0.0000 | **−46.73** |
| `ls_dispatch.ld_dispatch:u` | −0.499 % | 0.042 | 20/20 | 0.0000 | −11.3 |
| `ls_dispatch.store_dispatch:u` | −0.806 % | 0.041 | 20/20 | 0.0000 | −7.3 |
| `L1-dcache-loads:u` | −0.452 % | 0.029 | 20/20 | 0.0000 | −13.4 |

**`final54` reproduces the arm `land54` exactly** — both read −0.769 %,
se 0.000, in the same panel. That is the "I measured what I landed" proof, and
it is needed because promotion changes the binary: `land54`'s `.text` md5 is
`fe788675`, `final54`'s is `1ce4526f`, since removing `#[cfg]` scaffolding
relocates panic `Location` constants (study 35 §5.1).

**Gates.** `All checks passed`; eval hash `2e1ede95…93cf79c4`; d13
**1 739 588**; `Bench: 587683`; **`perft_full_suite` PASSED** (mandatory — board
mutation is touched). Frame node total **5 001 562** on every one of the nine
binaries in this study. **No cycles claim** (§4).

For scale: study 52 landed −36.5 i/n, study 53 −39.34, **this one −46.73.**

### 0.1 ★ The fusion IS a win — and four-fifths of it is not the fusion

The commission's first deliverable was to re-derive study 35 §4's −11.36 i/n
against today's tree. Rebuilt, the arm measures **−25.5 i/n** (−0.419 %,
24/24, se 0.000). It is a win, and larger than study 35's number.

**But study 35's own arithmetic said it should be a LOSS, and its stated
mechanism is mostly wrong.** Two corrections, both reported first because both
were load-bearing:

**(a) The arm was priced against a base that no longer exists.** Study 35's
table is `base` 5659.760, `after` 5634.760 (−25.0), `hookc` 5648.400 (−11.36).
`after` is what landed. So on today's tree the honest prediction from study 35's
own numbers was `hookc − after` = **+13.64 i/n, a regression.** Measured, it is
**−25.5 i/n.** The prediction is wrong by 39 i/n and in the opposite direction.

**(b) The mechanism is single-call-site inlining, not fusion.** `hookc54`
introduces a new hook type, `TtPrefetchDirty`, used at exactly ONE call site —
the ProbCut verification probe stays on plain `TtPrefetch`. That makes
`make_move_with::<H>` a single-caller `internal` function, and LLVM inlines all
**2466 bytes** of it into `do_move` (which grows 0x3de → 0xdbc; the
`make_move_with::<TtPrefetchDirty>` symbol does not exist in the binary).

`inl54` isolates that: a hook type **behaviourally identical to `TtPrefetch`**,
differing only in being used at one call site. No fusion, no `WANTS_DIRTY`, no
new `DirtyPiece` constructor. It reproduces the inlining (verified: no
`make_move_with::<TtPrefetchOnly>` symbol) — and it is worth almost all of it:

| arm, `panel-main.txt` n = 24 | instr % | **i/n** | share of `hookc54` |
|---|---:|---:|---:|
| `hookc54` (fusion + inlining) | −0.419 | **−25.5** | 100 % |
| `inl54` (inlining ALONE) | −0.342 | **−20.8** | **81.6 %** |
| ⇒ the fusion itself | −0.077 | **−4.7** | 18.4 % |

Both 24/24 with se 0.000. **Study 35 attributed this arm's whole effect to
deriving the record from registers; on today's tree that mechanism is worth
−4.7 i/n and a codegen accident is worth −20.8.**

Study 35's finding 3 is also refuted in passing. It said `hookc` "deletes
instructions but not loads" (−0.04 %, "indistinguishable from a byte-identical
null"), and used that to prefer `after`. Today `hookc54` moves
`ls_dispatch.ld_dispatch` by **−0.854 % at 24/24** — twenty times its own null.

**The judgement call, stated as one.** `inl54` gets 81.6 % of the win for ~15
lines and no new derivation. The extra −4.7 i/n costs a third `DirtyPiece`
constructor, a `WANTS_DIRTY` const on the hook trait, and a PSQT search that
builds a record it never reads (the const cannot see `nnue_active()`). I landed
the pair because `instructions:u` is the counter study 52 §4 licenses and −25.5
beats −20.8 on it — but a reader who weights simplicity above 4.7 i/n should
land `inl54` alone, and the numbers to do that are in this section.

### 0.2 ★ `init_quiets`: the idea won, the FIRST SHAPE OF IT LOST, and the gap is 30 i/n

Commissioned item 2. `init_quiets` reads the mailbox once per scored quiet
(3.93 quiets/node) and derives `pt`, `&rows[pt]` and `pc.index()` from it —
but `generate_all` emits quiets in piece-type order, so all of that is constant
across runs. Hoisting it is the arm.

**Built the obvious way, it is a clear loss:**

| arm | instr % | i/n | ld % | ld/n |
|---|---:|---:|---:|---:|
| `qtrun54` — nested `while i < n` over runs | **+0.155** | **+9.4** | +0.504 | +11.4 |
| `qtrunb54` — same idea, `list.iter()` kept | **−0.350** | **−21.3** | −0.087 | −2.0 |

Both 0/24 and 20/20 respectively, se 0.000. **The same idea is +9.4 or −21.3
depending on how the loop is spelled — a 30.7 i/n swing with identical
semantics.** The cause is visible in the source: the nested form indexes
`list[i]` twice per move (once as the element, once in the continuation test)
and each is a **bounds-checked** slice index, where `list.iter()` has none. It
paid two indexed loads and two compares to save one mailbox load.

This is study 13 §5.5's law again — the cost of a change is only weakly related
to the work the change does — and it is the reason the landed comment tells the
next person to re-measure rather than assume the idea carries.

### 0.3 ★ `sortclamp45` DOES NOT REPLICATE — retract the −2.002 %

The commission asked me to check whether study 45's `sortclamp45` still
reproduces and, if so, hand over a screen invocation. **It does not, and the
evidence was already on disk the night it was measured.**

The cited figure (−2.002 % total nodes, 51/29, sign p = 0.0183, median ratio
0.9745) is real and I reproduced it exactly. But the same worktree contains a
**second run, written 16 minutes later, at a different book offset (`o317`)** —
a replication the study never wrote up. Both re-run by me with study 45's own
read-only reducer:

```
bigq-batch-h16-d13-n4000        n=80   sc  -2.002%  51/29  p=0.0183  0.9745 [0.9599, 0.9976]
bigq-batch-h16-d13-n4000-o317   n=80   sc  -1.184%  43/37  p=0.5764  0.9889 [0.9725, 1.0061]
```

The replication's sign test is a coin flip and **its bootstrap 95 % CI crosses
1.0**. Study 45 also recorded that the same arm **changed sign** between the
correlated and independent versions of the instrument (+18.643 % → −6.545 %),
and that its cycles panel was discarded because the free-shuffle null was larger
than either arm it was meant to adjudicate.

**So no screen invocation is handed over, because starting a match on this would
be spending the box on a result whose own replication is a coin flip.** The
one-line patch still applies cleanly to today's `main` (`git diff 6ddbee28^
6ddbee28 -- src/movepick.rs` applies at offset +55) if the owner wants it
anyway; the honest prerequisite is a third and fourth book offset first, not
games.

### 0.4 ★ Study 53's counter disagreement is REAL, and reproduces to 0.1 loads/node

Commissioned item 3. Study 53 landed `gc53` + `magic53` together and reported
that `magic53` alone beat the combination on loads — −22.3 loads/node (10/10)
versus −10.6 (7/10, ns) — while instructions added to 0.1 i/node. It could not
settle whether that was a real interaction or a counter artifact.

`nogc54` reverts **just** `gc53` on the shipped tree, which makes today's `main`
the "combination" and `nogc54` the "`magic53` alone" arm. At n = 24:

| `nogc54` vs `base54` | median | se | neg/n | per node | study 53 predicted |
|---|---:|---:|---:|---:|---:|
| `instructions:u` | **+0.073 %** | 0.000 | 0/24 | **+4.4** | **+4.4** |
| `ls_dispatch.ld_dispatch:u` | **−0.515 %** | 0.042 | 24/24 | **−11.6 raw** | **−11.7** |
| `ls_dispatch.store_dispatch:u` | −1.198 % | 0.044 | 24/24 | −10.8 raw | — |

**Both of study 53's numbers reproduce, one exactly and one to 0.1 loads/node,
on a different night at 2.4× the sample size.** The disagreement is a property
of the binaries, not of the counter: `gc53` genuinely buys 4.4 instructions/node
and genuinely costs ~6.7 loads/node net of this panel's null (11.6 raw).

**What I could NOT do is break the tie.** Study 53 §8 item 1 asked for a
`cycles:u` null under 0.15 % so cycles could arbitrate. Mine is **−0.438 %,
se 0.349** — worse, not better, and in one panel the byte-identical null moved
cycles *significantly* (+0.553 %, 5/20, p = 0.0414). The tie stands open. A
static census found no mechanism either: `search::<NonPv>` has **identical**
static stack traffic in both binaries (241 stack loads, 110 stack stores), so
the "allocator fragility / extra spills" hypothesis is **not** supported by the
disassembly and I am not asserting it.

---

## 1. THE FUSION, AS BUILT

`DirtyPiece::from_actors(board, m, us, pc, captured, capsq)` builds the record
from what `make_move_with` step 3 already holds in registers. Two things make it
cheaper than either predecessor:

* **Nothing is read back.** `after_move` (what shipped) loads `state.moved` and
  `state.captured`; `from_move` (what shipped before that) loads
  `mailbox[from]`, `mailbox[to]` and `side_to_move`. `from_actors` loads none of
  them.
* **Two hot arms, not three.** Step 3 already resolves en passant into
  `(Some(their pawn), ep square)` so steps 5 and 10 need no special case, which
  collapses `MoveKind::EnPassant` and `MoveKind::Normal` into one straight-line
  arm — and deletes the `#[cold]` en-passant helper both predecessors needed.
  Castling and promotion stay `#[cold] #[inline(never)]`, because this body is
  inlined into `make_move_with`, whose prologue study 13 §5.1 measured spilling.

`const WANTS_DIRTY: bool` on `MakeMoveHooks` (default `false`) keeps `NoHooks` —
`Board::make_move`, perft, the classical path — and the ProbCut verification
probe monomorphized clean. The seam stays bench-neutral by construction, which
the unmoved `Bench: 587683` confirms.

### 1.1 The correctness argument, and the new oracle

`DirtyPiece` now has **three** independent derivations of the same fact:
`from_move` from the parent's mailbox, `after_move` from the child's
`StateInfo`, `from_actors` from step 3's registers. `from_move` is the oracle
for both of the others.

Added: **`from_actors_reproduces_from_move_for_every_move_kind`** — the twin of
study 35's test, but driven **through `make_move_with` with a recording hook**,
so what it checks is the record the search actually receives via Hook C rather
than a direct call. Same six roots, same depth-3 walk, same per-kind coverage
assertion, same whole-record `assert_eq!` (which pins emission ORDER, not just
the multiset — the accumulator folds these in sequence and the Finny backward
walk indexes `subs()[0]` positionally).

Empirically: node total 5 001 562, d13 1 739 588 and eval hash `2e1ede95…` are
identical on every binary in this study.

---

## 2. THE `init_quiets` RUN-HOISTING, AS BUILT

The shipped loop is unchanged in shape — `list.iter().enumerate()`, one
`score_quiet!` per move. What changed is that `run`, `row` and `pci` are
loop-carried, and the mailbox is consulted only when the from-square leaves the
current piece's bitboard:

```rust
if !run.contains(m.from()) {
    let pc = board.piece_on(m.from()).expect(...);
    run = board.pieces(pc.color, pc.piece_type);
    row = &rows[pc.piece_type as usize];
    pci = pc.index();
}
```

**It does not assume the grouping.** The run is discovered, not asserted: an
ungrouped list simply refills more often and degenerates to exactly the previous
cost. That matters — the grouping is `movegen`'s emission order, an
implementation detail no design doc pins, and not one this loop may silently
depend on. Emission order, scores and slot writes are untouched, so the tree is
bit-identical.

The per-quiet body was factored into a `macro_rules!` so the two loop shapes
under test shared one source. It stays after promotion because the body both
reads `self.quiet_prior` and calls `self.set_slot`, and any binding holding a
borrow of `*self` across it collides with the store — the constraint the
`policy` arm's existing comment already records.

### 2.1 What this does NOT do, and the ceiling that says so

It does not touch `partial_insertion_sort`, which study 34 measured at 2.00 % of
engine cycles and **61.52 % of `init_quiets`' mispredicts (8.46 % of ALL engine
mispredicts)**. Study 40 priced that class: at the measured **9.13 cycles per
mispredict** — not the ~18 everyone assumed — `init_quiets`' entire mispredict
share (13.75 %, 3.31/node) is **30.2 cycles/node = 0.74 % of cycles**, and
removing all of it must cost under **74 instructions/node** to break even. That
is the arithmetic the commission asked for, and it is why this study attacked
the scoring loop (3.25 % of cycles) rather than the sort.

---

## 3. THE PANELS AND THE NULLS

Harness `tools/perf54/exact54.sh` + `med54.py`, rebased verbatim from study 53 —
**blocked Latin square** (rotate within a block of NLAB, reshuffle the base
between blocks). Every panel carries `null54`, a byte-for-byte `cp` of
`base54`, riding in the rotation. Deltas are paired **within** round.

| panel | n | labels | file |
|---|---:|---|---|
| null establishment | 12 | base, null | `panel-null.txt` |
| first re-derivation | 20 | base, null, ctl, hookc | `panel.txt` |
| decomposition | 24 | base, null, hookc, inl, qtrun, nogc | `panel-main.txt` |
| confirmation | 20 (of 24) | base, null, hookc, qtrunb, land | `panel-conf.txt` |
| landing | 20 | base, null, land, final | `panel-final.txt` |

### 3.1 The inert control IS inert this time

Study 53's §2.2 could not read its single-arm column as a feature effect,
because its `ctl53` — the refactor with every feature OFF — was **−6.9 i/n on
its own** and had a different `.text` md5. Mine does not have that problem:
`ctl54`'s `.text` md5 is **`121e12b7`, identical to `base54`**, and it measures
**0.000 % on `instructions:u`** at 9/20. Its full sha256 differs (panic
`Location` line numbers again), which is precisely why `.text` md5 is the right
identity test in this crate. **Every marginal effect in this study is therefore
a straight arm-minus-base contrast, with no decomposition needed.**

### 3.2 `instructions:u` is exact; the load counters have a floor that MOVES

Byte-identical `null54`, across the three largest panels:

| counter | n=20 (`panel.txt`) | n=24 (`panel-main.txt`) | n=20 (`panel-final.txt`) |
|---|---:|---:|---:|
| `instructions:u` | 0.000 %, 9/20 | −0.000 %, 16/24 | −0.000 %, 14/20 |
| `ls_dispatch.ld_dispatch:u` | −0.209 %, 15/20, p=0.041 | **−0.219 %, 22/24, p=0.0000** | −0.028 %, 11/20, ns |
| `ls_dispatch.store_dispatch:u` | −0.242 %, 16/20, p=0.012 | **−0.210 %, 21/24, p=0.0003** | −0.028 %, 11/20, ns |
| `cycles:u` | −0.205 %, se 0.254 | −0.438 %, se 0.349 | **+0.553 %, 5/20, p=0.041** |

`instructions:u` is exact in every panel. **The load and store counters have a
floor that is significantly non-zero in two panels out of three and clean in the
third** — which independently confirms study 53 §2.5's observation on a
different night, and is why every load figure in §0.0 is quoted from
`panel-final.txt`, whose null is clean, and every load figure elsewhere is
quoted net of that panel's own null.

### 3.3 Additivity: instructions add EXACTLY, loads do not

`land54` was **built and measured**, not summed (study 53 §2.2's rule):

| `panel-conf.txt`, n = 20 | instr % | i/n | ld % | st % |
|---|---:|---:|---:|---:|
| `hookc54` | −0.419 | −25.5 | −0.901 | −0.967 |
| `qtrunb54` | −0.350 | −21.3 | −0.087 | −0.324 |
| **sum** | **−0.769** | **−46.8** | −0.988 | −1.291 |
| **`land54` measured** | **−0.769** | **−46.7** | **−0.641** | **−0.871** |

**Instructions add to the third decimal place.** Loads and stores do not — the
combination gives back ~7.8 loads/node and ~9.5 stores/node against the sum,
the same non-additivity study 53 §0.2 found in a different pair of arms. That is
now two independent sightings, and neither has a mechanism.

### 3.4 A reporting artifact in the analyzer, not in the data

`panel-conf.txt` prints mean positions of 1.92–2.08 rather than the ideal 2.00.
`med54.py` truncates to whole blocks (24 rounds → 20 used = 4 complete blocks,
which are balanced by construction) but computes the position mean over **all**
parsed rounds, including the 4 unused ones. The adjudicated rounds are balanced;
the printed mean is not the adjudicated set. Every other panel is a whole
multiple and prints exactly the ideal value.

---

## 4. NO CYCLES CLAIM

The byte-identical null read −0.111 % (se 0.337), −0.205 % (se 0.254),
−0.438 % (se 0.349), −0.346 % (se 0.522) and **+0.553 % (5/20, p = 0.0414)**
across this study's five panels. In the landing panel the null moved cycles
*significantly*. **The counter cannot resolve this and no cycles claim is made
for anything** — the same conclusion studies 35, 49, 50, 51, 52 and 53 each
reached with their own nulls.

Honesty note in the other direction, so nobody can say the favourable readings
were dropped: `land54`'s cycles median is −1.113 % (se 0.420, 13/20) in the
confirmation panel and `final54`'s is −0.381 % (se 0.284, 13/20) in the landing
panel. Both point the right way. **Neither is evidence.**

---

## 5. GATES

| gate | result |
|---|---|
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh` | **All checks passed** |
| wl50 eval hash | **`2e1ede9521624dd8…93cf79c4`** — unmoved |
| d13 node total | **1 739 588** — unmoved |
| `Bench:` trailer (depth-6 default build) | **587683** — unmoved |
| **D24 perft ritual** (board mutation touched) | **`perft_full_suite` PASSED** |
| frame node identity, all 9 binaries | **5 001 562** exact |
| d13 node identity, all arms | **1 739 588** exact |
| `final54` reproduces `land54` | **−0.769 % both**, se 0.000, same panel |
| byte-identical null, `instructions:u` | **0.000 %**, se 0.000, all panels |
| inert control `ctl54` `.text` md5 | **`121e12b7` = `base54`** |

No pin moves, and none should — both changes are behaviour-preserving, so a
moved pin would have been evidence that an equivalence was wrong.

---

## 6. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **Why `inl54` is worth −20.8 i/n.** I established *that* single-call-site
   inlining of `make_move_with` into `do_move` is the mechanism, and I built the
   control that isolates it. I did not decompose what the inlined form actually
   saves — argument marshalling, the `TtPrefetch` stack temp, cross-call
   register preservation, or something else entirely. A per-line op census
   (`tools/perf52/ibsline52.py` pointed at `do_move`) would settle it, and it is
   the same open item study 53 §8 item 2 already names.
2. **The `gc53` load/instruction tie is still open** (§0.4). It reproduces; it
   is not a counter artifact; `cycles:u` cannot arbitrate on this box and my
   null was worse than study 53's, not better. The static disassembly does *not*
   support the spill hypothesis, so the mechanism is unknown.
3. **Loads and stores are not additive across arms, twice now, with no
   mechanism** (§3.3).
4. **No PGO measurement.** Every number here is non-PGO. Study 35 §9 item 3
   predicted the `hookc`/`after` ordering would hold under PGO and never tested
   it; that prediction is still untested, and it now matters more, because the
   dominant mechanism turned out to be an inlining decision — exactly the kind
   of thing PGO re-litigates.
5. **No games.** Both changes are node-identical refactors, so games are not
   owed for correctness; they are owed for the speed→Elo conversion, at the
   standing ~120 Elo per doubling.

---

## 7. WHAT THE NEXT STUDY SHOULD RUN

1. **Take the PGO pair.** `base54` vs `final54`, both PGO'd, protocol in study
   52 §9.2 (panel the same source against itself first to bound retraining
   noise). This converts the largest inference in the series into a measurement
   and is now overdue across studies 52, 53 and 54.
2. **Census `do_move`'s body** — still study 49's genuine open item: 260.3
   i/node and 116.3 loads/node, never decomposed. It is now a *different*
   function (2466 bytes of `make_move_with` are inside it), so the census is
   both more necessary and more informative than when study 53 recommended it.
   **Do not price an arm off the 461 % stall figure** (study 53 §0.1b).
3. **`init_quiets`' scoring loop is still 3.25 % of cycles** and this study took
   only the mailbox and row derivation out of it. Study 34 §4.2 priced the ~11
   stack reloads of loop-invariant plane pointers per scored quiet and called
   them "not removable" on a 15-GPR argument that counts 12 live values — that
   claim has never been tested against a build.
4. **`sortclamp45` needs two more book offsets, not a match** (§0.3).
5. **Do not build on `ls_stlf`** (study 53 §0.4 — it fails its own null) and do
   not build `cont53` (study 53 §0.3 — it works, at +1.125 MiB per worker
   thread, breaching design/09 §8; that is the owner's call, not a perf trade).

---

## 8. REPRODUCTION

```bash
git checkout perf-study-54-fusion
bash tools/perf54/build54.sh base54          # or any labelled arm
bash tools/perf54/exact54.sh 24 base54 null54 final54
python3 tools/perf54/med54.py docs/research/perf-artifacts-54/panel.txt base54 null54 final54
CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh
cargo test --release -- --ignored perft_full_suite
```

Retained measurement feature: `nogc54` (§0.4), the instrument for study 53 §6
item 2.

### 8.1 The two deleted arms, preserved verbatim

`qtrun54` and `inl54` were deleted from the tree after measurement and are
**not** in this branch's history (the commit that would have carried them was
made against the wrong worktree and reverted — see §8.2). They are reproduced
here so both results stay checkable.

**`inl54`** — the decomposition control, worth **−20.8 i/n**, 81.6 % of
`hookc54`. Add this type and route `do_move`'s make through it instead of
`TtPrefetch`; change nothing else:

```rust
struct TtPrefetchOnly<'a> {
    tt: &'a TranspositionTable,
}

impl MakeMoveHooks for TtPrefetchOnly<'_> {
    #[inline]
    fn on_key(&mut self, key: Key) {
        self.tt.prefetch(key);
    }
}
```

**`qtrun54`** — the losing shape of §0.2, **+9.4 i/n**, replacing the scoring
loop in `MovePicker::init_quiets`:

```rust
let mut i = 0usize;
while i < n {
    let pc = board.piece_on(list[i].from()).expect("...");
    let pt = pc.piece_type;
    let row = &rows[pt as usize];
    let pci = pc.index();
    let run = board.pieces(pc.color, pt);
    loop {
        let m = list[i];
        score_quiet!(m, i, row, pci);
        i += 1;
        if i >= n || !run.contains(list[i].from()) {
            break;
        }
    }
}
```

The two bounds-checked `list[i]` reads per move — `let m = list[i]` and the
continuation test — are the whole difference from the landed shape.

### 8.2 A process error, recorded

One commit in this session was made against the wrong worktree. A command of
the form `cd <worktree> && <job> & disown; git add -A && git commit` backgrounds
the *entire* `cd && job` clause, so the `git` calls ran in the original cwd —
the main repo on branch `repr-gap-study` — and committed that branch's
unrelated WIP under this study's message. It was reverted with
`git reset --soft HEAD~1 && git reset`, restoring all three files to
modified-uncommitted exactly as found; nothing was lost. **Use `git -C <path>`
in any command line that also backgrounds something.**
