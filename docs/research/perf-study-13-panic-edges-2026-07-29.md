# ARCHON perf study #13 — the panic edges, and two mechanisms that were wrong

Date: 2026-07-29. Thirteenth commissioned study (11: `perf-study-11-fair-target-2026-07-29.md`,
12: `perf-study-12-fold-tile-2026-07-29.md`). Raw artifacts and instruments:
`perf-artifacts-13/`. Baseline: `main` at `0ca0c3c` (study 12 merged and gated).
Comparison target: **`engines/bin/stockfish-eca43a97`**.

---

## 0. HEADLINE

### 0.1 The ratio moved

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, two arms interleaved,
`taskset -c 4`, `nice -n 19`, 5 rounds, both PGO. Raw:
`counters-headline.txt`.

| per node | **ARCHON study 13** | ARCHON study 12 | **SF `eca43a97`** |
|---|---:|---:|---:|
| **instructions** | **6081.2** | 6114.6 | **4452.4** |
| **ratio vs `eca43a97`** | **1.3658×** | 1.3733× | — |
| branches | **652.4** | 667.6 | 362.7 |
| branch-misses | 20.8 | 20.6 | 19.7 |
| nodes | 15,001,568 | 15,001,568 | 15,008,494 |
| `instructions:u` spread | 0.0003% | 0.0390% | 0.0000% |

**Stockfish reproduces 4452.4 with 0.0000% spread** — the same control study 12
used, at the same value, so the two ARCHON numbers are in one frame.

**1.3733× → 1.3658×. −33.4 i/n, 2.0% of the 1,662 i/n gap, in four
node-identical changes.** Branches fell 15.2 per node. **1,628.8 i/n remain.**

### 0.2 What landed

Every arm is node-identical (5,001,562 on the probe tree, 15,001,568 on the
headline bench) and eval-hash-identical (`5ca978e012da1bb0`), and the gate's
`Bench: 587683` is unchanged.

| # | change | probe Δ | where |
|---|---|---:|---|
| 1 | policy head snapshotted per search instead of an `RwLock` read per node | −4.8 | `search/mod.rs` |
| 2 | `threat_map()` — one `ensure_threats` for the quiet scorer's five reads | **−19.0** | `board.rs`, `movepick.rs` |
| 3 | `king_square` without its two panic edges | **−13.6** | `board.rs` |
| 4 | `check_squares_all()` hoisted out of the per-quiet loop | −5.6 | `board.rs`, `movepick.rs` |
| | **total, probe frame** | **−43.0** | |
| | **total, PGO frame (the number that counts)** | **−33.4** | |

The probe overstated the PGO delta by 1.29× here, against study 12's ~2×.

### 0.3 What this study is actually about

**Two of the four hypotheses were wrong, and both were measured wrong rather
than argued wrong.** They are written up at the same length as the wins (§5),
because they close doors: the duplicate `DirtyPiece` derivation is *cheaper left
duplicated* (+38.2 i/n to fix), and **our 1.84× branch ratio against Stockfish is
not bounds checks** — deleting the densest bounds check in the engine cost
+43.8 i/n and moved branches by −0.2 per node.

What did work, twice, is a single shape: **a hot helper that is called N times
where once would do, or that carries a panic edge on a query the caller makes
tens of times per node.** Both wins are that. §8 item 1 is the biggest remaining
instance of it, and it is counted, not guessed: `Board::st()` runs **29.07 times
per node** through `Vec::last().expect(...)`.

---

## 1. METHOD, AND THE ONE THING THIS STUDY ADDS TO IT

Studies 8–12's rules were followed. One new gate, and it is the reason two of
this study's arms were thrown away instead of landed:

### 1.1 NULL FIRST — twice, and the second null is the new one

**Null A — rebuild determinism.** The same source, built and measured twice:

```
null-1   32,695,848,765   6537.1 i/n
null-2   32,695,922,205   6537.1 i/n     spread 0.00022%, ±0.015 i/n
```

**Null B — SOURCE PERTURBATION.** Null A only proves the toolchain is
deterministic. It says nothing about whether a *semantically inert* source
change moves the statistic — and at this level of optimisation that is the
question, because a single-site edit can move register allocation and inlining
far more than it moves the work. So: `AttackTables::{rook,bishop}_attacks` were
re-spelled into a three-line form with the same loads, the same arithmetic and
the same bounds check, and re-measured.

```
a0-INERT-accessor-respell   32,671,504,275   6532.3 i/n
a1 (same tree, one-line spelling difference)  32,671,559,510   6532.3 i/n
```

**±0.011 i/n.** The instrument survives its own null in both directions, so
every arm below is a real effect of that arm — including the two that made the
engine *slower*.

### 1.2 Instruments

* **`probe13fast.sh`** — screening. `CARGO_PROFILE_RELEASE_LTO=thin`
  / `CODEGEN_UNITS=4`, native, non-PGO, fixed 5,001,562-node tree, prints the
  wl50 eval hash and the node count so a tree or eval change cannot hide inside
  a speed claim. Study 12 §0.3's rule stands: **no probe delta may be quoted as
  a result**; only the PGO panel decides.
* **`pgo13.sh` + `pstat8.sh`** — the deliverable frame.
* **`counters13.patch`** — deterministic workload counters on **both** engines'
  shared quantities plus ARCHON-only call counts. Row and call counts are
  workload facts: immune to codegen, PGO, sampling and box load.
* **`blocks13.py`** — buckets two `perf report` flat profiles into comparable
  blocks from one table of regexes, so a block at parity cannot be quoted as a
  gap by accident (study 12 had to correct itself twice for exactly that).

### 1.3 The memory guard, corrected

`probe12fast.sh` refused to build whenever `free -g` reported `< 6`. `free -g`
**truncates**: 6134 MB reads as "5", and the guard refused builds with 6.0 GB
genuinely free. `probe13fast.sh` guards in megabytes (`< 4500`), which is the
same safety with honest units.

---

## 2. THE COUNTER PANEL — what actually runs per node

`counters.txt` / `counters-r2.txt`, from a build carrying relaxed-atomic
counters at each site (reverted before landing; the patch is the artifact).

| per node | count |
|---|---:|
| **`Board::st()`** | **29.0686** |
| **`Board::piece_on`** | **20.4402** |
| `AttackTables::rook_attacks` | 5.7053 |
| `AttackTables::bishop_attacks` | 5.2920 |
| `attacks::tables()` (a `LazyLock` deref) | 5.6239 |
| `set_check_info` | 1.0182 |
| `make_move` / `unmake_move` | 1.0059 |
| `ThreatPosition::from_board` | 1.0122 |
| `attackers_to` | 0.9713 |
| `attackers_to_exist` | 0.6657 |
| `between()` | 0.4365 |

Two of these decided the study.

**(a) `Board::st()` runs 29 times per node.** It is
`self.states.last().expect("state stack never empty")` — a length load, an
emptiness test, a panic edge, and only then the pointer arithmetic. Every state
accessor in the engine (`key`, `checkers`, `check_squares`, `blockers_for_king`,
`rule50`, `pawn_key`, …) goes through it.

**(b) 11.0 slider lookups per node**, against Stockfish's dual hyperbola
quintessence which returns rook *and* bishop from one 32-byte load
(`eca43a97:src/attacks.h`, `DualMagic::both_attacks_bb`). See §6 — porting it
would *raise* our instruction count, and the counter is how we know.

---

## 3. THE ARM LOG

Every measurement this study made, in order, in the fast probe frame
(`probes-fast.txt`). The two nulls bracket the arms.

```
null-1                       6537.1    baseline, main @ 0ca0c3c
null-2                       6537.1    rebuild null, +-0.015 i/n
a1-policyhead-snapshot       6532.3     -4.8   LANDED
a2-dirty-via-hookc           6573.8    +41.5   rejected (S5.1)
a2b-hookc-byvalue            6570.5    +38.2   rejected (S5.1)
a0-INERT-accessor-respell    6532.3    +-0.0   source-perturbation null
a3-attacks-table-mask        6576.1    +43.8   rejected (S5.2)
a4-threat-map-one-fill       6513.3    -19.0   LANDED
a5-king-square-no-panic-edge 6499.7    -13.6   LANDED
a6-check-squares-hoist       6494.1     -5.6   LANDED
```

Node count 5,001,562 and eval hash `5ca978e012da1bb0` on **every line**,
including the rejected ones — the two losses are correct code that is slower,
not broken code.

## 4. WHAT LANDED, AND WHY EACH ONE WORKED

### 4.1 The policy head's `RwLock`, taken once per search instead of once per node (−4.8)

`search::<NT>` called `crate::eval::nnue::active_policy_head()` at every
interior node with no TT move, and again wherever `CPL_ORD_W` is set. That
function is

```rust
*ACTIVE_POLICY_HEAD.read().unwrap_or_else(|e| e.into_inner())
```

— a process-wide `RwLock` read lock: an atomic read-modify-write to acquire, a
poison check, another to release, for a value that **cannot change during a
search**. Nets are installed once per `go`, by `refresh_eval_net`, and D47
policy is switched off in production anyway (no `.cpol` sidecar ⇒ `None`).

`SotaContext` already snapshots the same handle at net-set time
(`sota_context.rs:213`). This is that pattern applied to the search: a
`Worker::policy_head` field written in `refresh_eval_net` beside the nets.

−4.8 i/n is the smallest arm here and it is worth noting that the instruction
count *understates* it: two `lock cmpxchg`-class operations per node cost far
more in cycles than in instructions. It is quoted at what the metric says.

### 4.2 One `ensure_threats` instead of five (−19.0)

`MovePicker::init_quiets` builds its escape/entry tiers from the D30 threat map:

```rust
let by_pawn = board.piece_threats(PieceType::Pawn);
let by_minor = by_pawn | board.piece_threats(Knight) | board.piece_threats(Bishop);
let by_rook  = by_minor | board.piece_threats(Rook);
…
let undefended = board.occupancy(them) & !board.all_threats();
```

Five accessors, **five `ensure_threats()` calls**, of which four can only ever
observe a cache the first one already filled. `ensure_threats` is not inlined,
and each call re-materialises `self.st()` — a `Vec::last()` with its own length
load, emptiness test and panic edge — then re-reads and re-compares the `Cell`
key.

`Board::threat_map()` returns `(all, by_type)` behind one fill check. Node- and
eval-identical: same sets, same source, same order.

**−19.0 i/n and −3.5 branches per node**, the study's largest arm.

### 4.3 `king_square` without its two panic edges (−13.6)

```rust
self.pieces(color, PieceType::King).lsb().expect("king must exist")
```

states the invariant twice for the optimiser to disbelieve twice: `lsb()` tests
the board for zero and wraps the result in an `Option`, then `expect` tests the
`Option` again. Two panic edges on a query that `set_check_info` alone makes
three times per node, and which also runs in movegen, the HalfKA refresh, the
threat resolver and the feature indexers.

`trailing_zeros()` of a non-zero `u64` is already in `0..64`. Masking with 63
says so to LLVM — which additionally folds away `Square::new`'s own
`assert!(index < 64)` — and the `debug_assert!(bb != 0)` is where the invariant
now lives, still checked at every node of every debug search.

**−13.6 i/n and −12.5 branches per node.** The branch figure is the one that
identifies the mechanism: ~12 branch-pairs per node disappeared, which is the
call rate this accessor has.

### 4.4 The check-square block, hoisted out of the per-quiet loop (−5.6)

The quiet scorer's check bonus reads `board.check_squares(pt)` once per scored
quiet — ~30 per node — with `pt` varying per move, so it cannot be hoisted by
hand at the call site, and LLVM only hoists the underlying `st()` when it can
prove nothing in the loop body invalidated it. `Board::check_squares_all()`
returns the whole 48-byte block, so the hoist is a property of the source.

−5.6 i/n. Small, and included because it is the same shape as §4.2 at a
different call rate, which is the evidence that the shape generalises.

---

## 5. WHAT DID NOT LAND — two mechanisms, both measured, both wrong

Both arms below are **correct** code: node-identical (5,001,562) and
eval-hash-identical (`5ca978e012da1bb0`). They are simply slower, and the null
of §1.1 is what makes that statement safe to make.

### 5.1 The duplicate `DirtyPiece` derivation — **+38.2 i/n**

`Worker::do_move` builds the accumulator's placement record with
`DirtyPiece::from_move(&self.board, m)` *before* the make. That call re-derives
the mover, the victim, the en-passant capture square and the castling rook
geometry — every one of which `make_move_with` computes a few instructions later
for its own Zobrist work. It measured as its own **53 i/n** symbol in study 12's
stage-split profile, and Stockfish fills its `DirtyPiece&` out-parameter inline
in `do_move` (`eca43a97:src/position.cpp`, the `dp.pc = pc; dp.from = from; …`
assignments). The seam for doing the same already existed and had never been
wired: `MakeMoveHooks::on_dirty`, Hook C, whose doc comment says *"the call site
lands with the search wiring in step 4"* — it never did.

So it was wired: an associated `const WANTS_DIRTY: bool` (so `NoHooks`, hence
perft and `Board::make_move`, monomorphizes the whole construction away), the
record built from `make_move_with`'s own locals in `from_move`'s exact emission
order, and `Worker::do_move` reading it back out of the hook.

**+41.5 i/n.** A second variant — payload passed by value instead of by
reference, and the castling rook geometry re-derived inside the guarded block
instead of carried in an `Option` local across the whole body — recovered 3.3 of
it: **+38.2 i/n**, still a loss.

The mechanism was real and the fix still made things worse: `make_move_with` is
a 650-instruction function that is already spilling, and the marginal cost of
materialising a `DirtyPiece` *there* exceeds the cost of deriving it from
scratch outside, where the register pressure is low. **The flat profile's 53 i/n
for `DirtyPiece::from_move` was not waste — it was work that has to happen
somewhere, and outside is the cheap place.** Reverted; the reason now sits in a
comment at the call site so study 14 does not re-derive it.

### 5.2 The slider table's bounds check — **+43.8 i/n**, and the thesis behind it is dead

Our headline branch count is **1.84× Stockfish's** (667.6 vs 362.7 per node,
study 12 §0.1). The obvious reading is Rust bounds checks, and the packed magic
table is the densest candidate: `self.table[magic.index(occ)]` on a
`Box<[Bitboard]>`, 11.0 times per node, each paying a length load, a `cmp` and a
`jae`. The index is in range **by construction** (the per-square blocks tile the
array and `init_magics` verifies every slot against the naive oracle), so the
check is provably dead.

Arm: allocate the table at `TABLE_SIZE.next_power_of_two()` (131,072 entries,
183 KiB of unreachable zeros) as a fixed-size `Box<[Bitboard; TABLE_CAP]>` and
range-limit the index with `& (TABLE_CAP - 1)` — the same idiom the threat
resolver's `& (DELTA_CAP - 1)` and the Finny diff's `& (CAP - 1)` already use.

**+43.8 i/n, and branches moved by −0.2 per node.**

That −0.2 is the finding. **The checks were not executing**, so there was
nothing to delete and the `and` was pure addition. A static count over the whole
disassembly confirms it independently: the square-range check `cmp $0x3f` occurs
**once or twice per hot function** and **zero times** in `see_ge`, `legal`,
`gives_check` and `init_quiets` — LLVM already knows `Move::from()`/`to()` are
masked to six bits, and CSEs the one check it cannot avoid across every
square-indexed table in the function.

**So our 1.84× branch ratio is NOT bounds checks.** Two independent
measurements say so. Study 14 should not spend a session on `get_unchecked`,
masking idioms, or a `Square` newtype with a narrower range: that lever has been
priced at zero. The branches are genuine control flow — `record_square_touch`
alone carries 127 conditional branches, `at_full` 139, `init_quiets` 50.

---

## 6. WHAT THE REFERENCE SAYS, AND WHY WE ARE NOT COPYING IT

Stockfish at `eca43a97` **no longer uses magic bitboards on x86**. With AVX2 it
compiles `USE_DUAL_HYPERBOLA_QUINT` (`src/attacks.h`): a 32-byte-aligned
`DualMagic` per square whose `both_attacks_bb` computes rook *and* bishop
attacks together — four masks in one `_mm256_load_si256`, one hyperbola-
quintessence step across all four lanes with a byte-swap standing in for the bit
reversal, an `or`/`extract` pair to split the results, and an 8-bit rank-attack
lookup for the one direction hyperbola cannot serve. The whole structure is
~4 KB plus a 2 KB rank table; **ours is an 841 KB packed magic array**, so every
slider query of ours is a guaranteed L2/L3 miss where theirs is L1-resident.

It is tempting, and it is the wrong trade **for this metric**. `both_attacks_bb`
is roughly 15 instructions for both piece types; our two magic lookups are ~5
instructions each, so ten. At the measured **11.0 slider lookups per node** a
port would *add* on the order of 30–50 i/n while removing cache misses we are
not being scored on. It is a **cycles** lever, not an instructions lever, and it
belongs to whoever is optimising NPS rather than the instruction ratio — with
the note that our headline is already `instructions:u`, so the two goals have
diverged here for the first time.

---

## 7. THE BLOCK TABLE, RE-DERIVED IN ONE FRAME

`blocks13.py` over study 12's two **stage-split** profiles (ARCHON at 6362 i/n,
SF at 5090 i/n — both calibrated off their own `search<NonPv>` symbol, which
study 12 priced at 606 and 617 respectively). Split frames are not comparable in
absolute terms; block *ordering* and block *differences* are.

| block | ARCHON | SF | delta |
|---|---:|---:|---:|
| search | 690 | 672 | +17 |
| movepick | 740 | 734 | +6 |
| eval head | 572 | 572 | **0** |
| accumulator fold | 1232 | 1099 | +133 |
| threat generation | 718 | 686 | +33 |
| eval glue (`SotaContext::push`/`evaluate`, `EvalContext`) | 376 | 171 | **+205** |
| **make / unmake** | 661 | 302 | **+359** |
| board queries | 493 | 380 | +113 |
| movegen | 110 | 47 | +63 |
| histories | 331 | 230 | +101 |
| tt | 56 | 81 | −25 |
| unbucketed | 238 | 23 | +215 |
| **TOTAL** | **6307** | **5059** | **+1248** |

Three of these are new information:

* **Threat generation is at parity (+33), not a defect.** Our lazy regenerate-
  at-fold design (`at_full` 363 + `record_square_touch` 140 + `record_targets`
  ~95 = ~600) costs *less* than Stockfish's eager incremental maintenance
  (`update_piece_threats<true>` 389 + `<false>` 120 + `append_changed_indices`
  177 = 686), because we only pay it on the 43% of nodes that evaluate. Study
  12's §11 item 3 sized `at_full` at "≤ +400" of headroom; the split says the
  headroom is ~0 and the 954 i/n of the unsplit profile is mostly the inlined
  i8 fold (513 when split out).
* **The accumulator fold is at its arithmetic floor.** Provable count from the
  disassembly and the row counters: `80 + 80·rows` instructions per pass, 6.8
  rows and 2.30 passes per node ⇒ ~732 i/n of unavoidable arithmetic, against
  SF's ~584 at 1024 lanes and 7.95 rows. The whole difference is width: at 1024
  we would be at 581 against their 584. **There is no code lever left here** —
  §8 item 6.
* **`make/unmake` (+359) and `eval glue` (+205) are where the gap actually is**,
  and between them they are 45% of the whole 1,248 i/n of split-frame excess.

And one row nobody has looked at: **`unbucketed` is +215**, i.e. ARCHON has 238
i/n of symbols that fit none of the twelve blocks against Stockfish's 23. Its
largest members are `l2_tail` 37 (the L2→L3 head tail, arguably "eval head"),
`Worker::stat_score` 27, **`active_policy_head` 25** (deleted by §4.1),
`Board::move_piece` 22 — *an out-of-line call inside `make_move_with`* — and
~68 i/n spread over four unsymbolised addresses in one 16-byte range, which is
almost certainly a libc `memmove`/`memcpy` variant. **Nobody has established
what is calling it.** That is a cheap and completely unexplored 68 i/n: build
with `-C force-frame-pointers=yes` (the `profiling` profile already does) and
take a call-graph profile of that address.

---

## 8. RANKED REMAINDER — handoff to study 14

**1,628.8 i/n remain**, of which ~327 is the 1280-vs-1024 accumulator width (a
retrain, not a code change — study 12 §11 item 9, unchanged here) and, by §7,
**~0 is the accumulator fold, the eval head, the threat generator, movepick or
the search body**. The code gap that is actually addressable now lives in
`make/unmake` (+359), `eval glue` (+205), `board queries` (+113), `histories`
(+101) and `movegen` (+63).

**Read §5 first.** Two of this study's four hypotheses were wrong in a way that
cost a build cycle each, and both are the kind that look obvious on paper.

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **`Board::st()` — 29.07 calls/node through `Vec::last().expect(...)`** | ~30–60 | **high — counted** | see below |
| 2 | **eval glue: `SotaContext::push` 115 + `push_sota_dirty` 139** | +205 vs SF's 171 | medium | 5 `Option` checks for absent side-nets per push, and `ThreatPosition::from_board` (1.01/node, ~25 ops + a 74-byte store) which could be derived from the parent snapshot by ≤4 `toggle`s |
| 3 | make/unmake's remaining +359 after item 1 | ? | medium | `set_check_info` is 59 i/n against SF's 33 for identical code; `move_piece` is a 22 i/n **out-of-line call** per make |
| 4 | fold-kernel row-loop control | ~20–40 | medium | 6 instructions per row per tile (`mov`/`lea`/`shl`/`add`/`cmp`/`jne`), 6.8 rows × 2 tiles = 82 i/n; resolve the byte offset at resolve time so the kernel's list is pre-scaled |
| 5 | `attacks::tables()` — 5.62 `LazyLock` derefs/node | ~17 | high | a `const`-evaluated table removes the check *and* the GOT load; ~5 M const-eval steps, so the risk is build time, not correctness |
| 5b | **the unsymbolised ~68 i/n of libc `memmove`** (§7) | ~68 | **unknown — nobody has looked** | one call-graph profile on the `profiling` profile; the caller is not in any flat table anyone has produced |
| 5c | `Board::move_piece` is a 22 i/n **out-of-line call** per make | ~15 | medium | it is the make/unmake hot path's innermost primitive and it is not being inlined; check why (`#[inline]` present? cold-call heuristics from the panic edges in `put_piece`/`remove_piece`?) |
| 6 | ~~accumulator width / fold~~ | **0 as code** | **closed** | §7 — at its arithmetic floor; the residual is 1280 vs 1024, a retrain |
| 7 | ~~threat generation~~ | **+33** | **closed** | §7 — cheaper than Stockfish's per-node equivalent |
| 8 | ~~bounds checks / the 1.84× branch ratio~~ | **0** | **REFUTED** | §5.2, two independent measurements |
| 9 | ~~`DirtyPiece::from_move` duplication~~ | **+38** | **REFUTED** | §5.1 |
| 10 | ~~dual hyperbola quintessence~~ | **negative for i/n** | **priced** | §6 — a cycles lever, not an instructions lever |

**Item 1, in full, because it is the ranked top and it is blocked on one thing.**
`st()` is 29.07 calls/node, and the shape that makes it expensive is
`Vec::last()`: a length load, a zero test and a panic edge before any useful
work. The fix is the standard one — `states: Box<[StateInfo; STATE_CAP]>` with a
`top: usize` cursor and `STATE_CAP` a power of two so the index is a mask — and
it also deletes the provably dead `states.push(StateInfo::EMPTY)` (three `zmm`
loads and three `zmm` stores of a 168-byte constant that is fully overwritten,
confirmed in the disassembly at `make_move_with+0x9e`) plus the `mem::take`
round trip.

**What blocks it**: `make_move_with` holds `st: &mut StateInfo` across calls that
need `&mut self` (`remove_piece`, `put_piece`, `move_piece`) and `&self`
(`set_check_info`, `attackers_to`). Today that is legal only because the whole
`Vec` is moved out with `mem::take` (a `Vec` has a cheap `Default`; a
`Box<[T; N]>` does not). Study 14 needs one of: (a) a `Placement<'_>` helper
struct holding disjoint `&mut` borrows of `pieces`/`color_occupancy`/
`all_occupancy`/`mailbox`, with the three placement mutators moved onto it — the
idiomatic Rust answer to a partial borrow, and it makes the borrow split
explicit instead of a `mem::take` trick; or (b) `set_check_info` returning its
ten bitboards by value so no `&mut StateInfo` needs to live across a `&self`
call. **This is a 1–2 hour refactor of the most D24-critical function in the
engine and it should be budgeted as one, not slipped in.**

**And a method warning that outranks all of the above.** Single-site changes in
this codebase move the statistic by **±40 i/n through codegen alone** — arm 5.2
added one `and` per lookup and cost 43.8. The rebuild null is ±0.015 and the
inert-respelling null is ±0.011, so this is not noise: it is real, arm-caused,
and *not attributable to the mechanism the arm names*. The practical
consequence: **price the arm, never the mechanism.** An arm that wins is worth
landing whether or not the story about why is true; an arm that loses is dead
even when the story is airtight. Study 12's five landed arms and this study's
four both fit that rule; the two refuted here are what happens when you do not.

---

## 9. GATE

`gate-s13.log`.

* `bash tools/check.sh` — **`Bench: 587683`**, unchanged. Arms 2 and 4 are in
  `movepick.rs` and `board.rs`, both of which the default-feature build
  compiles, so this is a real node-identity check on them and on `king_square`.
* `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
  (604 lib tests, 0 failed), the release perft suite (`Nodes searched: 1839507`)
  and the two-run deterministic bench-perft fingerprint.
* The D24 perft ritual applies (`king_square` is read by movegen and by the
  check bookkeeping) and passes as part of check.sh.
* **`CHECK_NNUE_GATE=1` was run even though the diff touches nothing under
  `src/eval/nnue/`** — `king_square` feeds the HalfKA king bucket, the Finny
  cache key and the threat resolver's orientation, so the bench pin alone would
  not have seen a mistake in it. Both NNUE pins reproduce:
  * eval hash `5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38`
  * d13 nodes **1922947**

**The gate triple — `Bench: 587683`, the eval hash, and d13 `1922947` — all
unchanged IS the node-identity proof for the whole landed set.**

### 9.1 Each landed change has a test that can fail, and it was verified by mutation

| change | test | mutation that makes it fail |
|---|---|---|
| `threat_map` | `threat_map_on_a_cold_cache_agrees_with_the_per_type_accessors` | drop the `ensure_threats()` ⇒ reads the cold all-zero cache |
| `king_square` | `king_square_is_exact_across_the_index_range` | `& 62` instead of `& 63` (h8/a1 corners are in the fixtures) |
| `check_squares_all` | `check_squares_all_agrees_with_the_per_type_accessor` | return `[EMPTY; 6]` |

All three mutations were applied and **all three tests failed**, then the
mutations were reverted and the suite re-run clean. The first test's boards are
deliberately FRESH — the existing `threat_map_matches_the_attackers_to_oracle`
would have passed a `threat_map` with no fill, because it warms the cache first.

---

## 10. THREATS TO VALIDITY

1. **Probe deltas overstate PGO deltas by ~2×** (study 12 §0.3). Every per-arm
   figure in §3 is probe-frame; only the total is PGO-measured.
2. **§7's block table mixes two split builds** with different `noinline` sets and
   different compilers. Block ordering and differences are safe; absolute i/n
   are ±.
3. **Nothing here was Elo-tested**, and there is nothing to test: every landed
   arm is node-identical (5,001,562 on the probe tree) and eval-bit-identical
   (`5ca978e012da1bb0`), so the engine plays exactly the same moves.
4. **The counter build perturbs inlining** — an atomic increment inside `st()`
   prevents LLVM from hoisting it out of loops, so 29.07 is the number of
   *calls*, not of executed materialisations. The instruction cost of item 1 is
   therefore an upper bound until it is measured by an arm.
5. **Box state**: a D48Y training run shared the box for the whole session
   (GPU at 91%, ~7.1 GB RSS). Everything here rides on `instructions:u` over
   fixed-node trees and on deterministic counters, both contention-immune; no
   wall-clock or NPS number is quoted as a claim.

---

## 11. ARTIFACTS (`perf-artifacts-13/`)

* `probes-fast.txt` — every arm with its eval hash and node count, including
  both nulls and both refuted arms.
* `counters.txt`, `counters-r2.txt` — the §2 deterministic panels.
* `counters-headline.txt` — the §0.1 two-arm PGO panel, 5 rounds, with the
  parsed ratio table appended.
* `gate-s13.log` — both gate runs (default and `CHECK_NNUE_GATE=1`).
* `counters13.patch` — the counter instrumentation (reverted).
* `scripts/probe13.sh`, `probe13fast.sh`, `pgo13.sh`, `null13.sh` — new.
* `scripts/blocks13.py` — the block bucketer; its output is `blocks.txt` (§7).
* `scripts/sfshow.sh` — pinned-revision reader for the reference clone.
* `scripts/pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`, `opcount11.py` —
  inherited.
