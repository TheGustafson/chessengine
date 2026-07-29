# ARCHON perf study #10 — the 1.10× target, and the floor that stops it

Date: 2026-07-28. Tenth commissioned study in the chain (8:
`perf-study-8-nps-parity-2026-07-28.md`, 9: `perf-study-9-parity-items-2026-07-28.md`).
Raw artifacts and instruments: `perf-artifacts-10/`. Branch `perf-study-10`,
based on `main` at the study-9 merge. Deliverable binary:
**`engines/bin/archon-s10-final`** (PGO, `sota,arch-d48x`, canonical net,
sha256 `fc73c9a7…6854a`). Baseline for every comparison:
`engines/bin/archon-s9-final`.

**Gates.** Both landed commits pass `CHECK_NNUE_GATE=1 bash tools/check.sh`
with **`Bench: 587683`**, wl50 eval hash **`5ca978e0…8b38`** and d13 nodes
**1,922,947** all unchanged, plus the full D24 perft tier. No pin moved,
because nothing here perturbs the eval.

**The mandate: reach ≤1.10× SF18 on instructions per node — remove 1,667 i/n,
26% of the engine, roughly 2.4× everything studies 8 and 9 landed together.
It was not reached, and this study's principal deliverable is the per-item
accounting that says where those 1,667 i/n are and why most of them are not
ours to remove.**

Two changes landed (−0.977% on the PGO panel, both bit-exact and
node-identical), one was implemented and killed by its own number, one was
measured and declined on that measurement, and **the largest structural item
the mandate named was tested and its premise is false**.

---

## 0. HEADLINE

### 0.1 The matched panel

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, three arms interleaved,
`taskset -c 4`, `nice -n 19`, 6 rounds, both ARCHON arms PGO. Raw:
`counters-final.txt`. Box during the panel: load 3.7–8.3, 3–5 GB free.

| per node | baseline (`archon-s9-final`) | **final (`archon-s10-final`)** | SF18 | 1.10× target |
|---|---:|---:|---:|---:|
| **instructions** | 6,429.9 | **6,367.1 (−0.977%)** | 4,329.6 | **4,762.6** |
| **ratio vs SF18** | 1.4851× | **1.4706×** | — | 1.100× |
| branches | 755.2 | **734.2 (−2.78%)** | 394.7 | — |
| branch-misses | 22.3 | 22.2 | 26.3 | — |
| nodes | 15,001,568 | 15,001,568 | 15,010,238 | — |
| `instructions:u` spread | 0.0033% | 0.0034% | 0.0000% | — |

Both ARCHON arms searched a **byte-identical 15,001,568-node tree**, so this
is a pure workload-controlled comparison. The baseline reproduced study 9's
6,429.9 exactly, and SF18's 4,329.6 exactly — the sixth independent session to
land SF18 on that figure.

**Distance to the bar: 1,604.5 i/n remain. This study closed 62.8 of the
1,667.3 required — 3.8% of it.**

The cycles and NPS columns are omitted from every conclusion: the box carried
another agent's multi-engine gauntlet all session (load 3.7–10, 1–5 GB free,
swap full at times), and the per-round paired NPS ratio ranged 0.527–0.755 on
byte-identical binaries searching byte-identical trees. `instructions:u` on
the same rounds reproduced at 0.003%.

### 0.2 Item by item

| # | item | verdict | the number |
|---|---|---|---|
| 1 | Panic edges in the threat resolver's per-event loop | **LANDED** | **−0.805%** (56.4 i/n). Five conditional branches to panic landing pads removed from a ~40-instruction loop that runs 4.4×/node. Found by a new instrument (§2.1). |
| 2 | HalfKA Finny refresh — hoist the row base, mask the cursor | **LANDED** | **−0.053%** (3.7 i/n) (§2.2). |
| 3 | HalfKA Finny refresh — iterate bitboard rows instead of `PieceType::ALL` | **KILLED** | **+0.266%** measured alone. The stack-materialised `ALL` array it deletes is cheaper than the aliasing facts the `iter_mut()` zip costs (§3.1). |
| 4 | **Incremental threat maintenance** (the mandate's largest named structural item) | **PREMISE KILLED** | The threat FOLD is at parity. `vpmovsxbw` — the i8→i16 widening only the FullThreats fold performs — is **216.5 i/n in ARCHON against 219.7 i/n in SF18** (§3.2). SF does not fold fewer threat rows than we do; it folds slightly more. |
| 5 | `set_check_info` laziness (priced 106 i/n by study 9, never attempted) | **MEASURED, DECLINED** | The read rate the mandate asked for: **1.0125 fills/node against 0.5103 move-generating nodes — 49.6% of fills are thrown away**, so the ceiling is **39 i/n (0.56%)**, against a movegen-contract change (§3.3). |
| 6 | `init_quiets`' remaining panic edges (13.71% of the symbol by annotate weight) | **NOT ATTEMPTED** | Study 8 §7.1 measured exactly this class at **−0.045%**; the weight is skid. The candidate list is not a ranking (§3.4). |
| 7 | **SF18's adaptive eval cost** | **PRICED** | SF18 runs a 128-wide net on lopsided positions and pays for its 1024-wide net only near equality. **23–37% of its evaluated nodes take the cheap path**, so **325–740 of the 1,667 i/n I was asked to remove is a feature we lack, not inefficiency we carry** (§4.1). |
| 8 | The same gate for ARCHON | **PRICED, RETRAIN-DEPENDENT** | At SF18's own threshold, **21.6% of our evaluated nodes qualify**. Ceiling **157–732 i/n**; the honest figure is near the low end, and the substitute net does not exist (§4.2). |

### 0.3 The finding that reframes the mandate

**Our threat channel's excess over SF18 is entirely scalar plumbing, not the
fold.** The fold is at parity (216.5 vs 219.7 i/n of `vpmovsxbw`). What is
2× SF's is the event generation, resolution and snapshotting around it —
which is why the one change that paid this study was in the resolver, and why
"make the threat channel incremental like SF's" cannot pay: SF's is not more
incremental than ours, and its fold is not cheaper.

This is the third consecutive study to kill a threat-channel structural
premise with a measurement — study 7 killed "1.78× / SF has no threat
channel", study 9 killed "unify the two threat systems", and this one kills
"make it incremental". The channel is expensive because it is a 60,720-dim
relational feature set folded into a 1280-wide accumulator, and both engines
pay for that in the same coin.

---

## 1. METHOD

Studies 8 and 9's rules were followed. Two additions earned their place:

* **A counting instrument before any candidate.** `src/perfcount.rs` (24
  atomic counters, added, measured, and reverted — the patch is
  `counters.patch`) put a number on every rate the eval layer runs at
  (§1.1). Every candidate in this study was sized as *loop body × trip count*
  from those rates, and the two that survived were the two whose count said
  they were real.
* **A panic-edge detector that can see this codebase's bounds checks.**
  Study 8's `bcheck8.py` classifies a bounds check by the jump's **target
  symbol**, and LLVM sinks these landing pads into the cold tail of the
  enclosing hot function — so `at_full`'s five per-event edges were invisible
  to it. `scripts/panicedge10.py` disassembles once, marks every basic block
  that calls a panic entry point, and reports conditional jumps into them by
  enclosing symbol (§2.1).

**Instructions carry every claim**, and per-change deltas come from **non-PGO
probe pairs** (identical source, identical flags ⇒ identical binary ⇒ the
measurement is exact, not statistical). PGO is used for the headline panel
only.

**NULL FIRST: the PGO pipeline's build-to-build variance is exactly zero.**
A concern was raised mid-session that two PGO builds of identical source
differ by ~1.7% in NPS, which would exceed every item in this study. So the
null was run before the arm: `tools/build_pgo.sh sota,arch-d48x` was invoked
**twice** on this branch's HEAD, minutes apart, at different system loads.

```
fc73c9a7c2004cd24ad814fb365280bf7390d491a3df6baffc5549f35cc6854a  archon-s10-final
fc73c9a7c2004cd24ad814fb365280bf7390d491a3df6baffc5549f35cc6854a  archon-s10-null
```

**Byte-identical.** The training workload is `bench … depth 8` plus two
fixed-`depth` UCI searches — all node-deterministic — so the merged profile
is deterministic and so is the codegen. **This pipeline contributes no
variance to an instruction count**, and the headline panel's resolution is
limited only by `instructions:u` spread (0.003%). The second binary was
deleted as a duplicate; the hash above is the record. (This does *not*
contradict the 1.7% figure, which compared two builds whose **sources
differed** by an added tunable, on a contended box, by NPS.)

**Two measurement rules were imposed mid-session and are recorded because they
change what earlier numbers mean:**

1. **Never compare NPS at fixed nodes across an eval-changing arm.** The two
   binaries then search different trees, so it is not a like-for-like
   workload. Nothing in this study changes the eval — both landed arms are
   eval bit-exact and node-identical — so the rule costs this study nothing,
   but it invalidates fixed-node NPS comparisons across studies 8's items 5
   and 6.
2. **Instructions/node converts to wall-clock depth at roughly 0.45×, not
   1:1** (measured elsewhere this session: studies 8+9's −9.65% instructions
   bought +0.110 ± 0.235 ply at equal time, ≈ +3.8% effective speed at the
   measured 2.02 ply/doubling). **This study's landed −0.977% is therefore
   worth ≈ +0.44% effective speed ⇒ ≈ +0.8 Elo at 120 Elo/doubling.** No match
   was run on it: at that size the expected depth delta is ~0.01 ply, two
   orders of magnitude inside the CI of an affordable match, so a match could
   not resolve it and would only have consumed a contended box.

### 1.1 The rates everything else is derived from

`perfcount`, 1,500,780 nodes, `bench 16 1 30000 wl50 nodes` with the canonical
net. Raw: `counters-perfcount.txt`.

| counter | per node | reads as |
|---|---:|---|
| `EVALS` | 0.4315 | net evaluations |
| `MAKE_MOVE` / `NULL_MOVE` | 1.0038 / 0.0087 | ⇒ **1.0125 `set_check_info` fills** |
| `MP_NEW` / `MP_NEW_PROBCUT` | 0.4623 / 0.0480 | ⇒ **0.5103 nodes generate moves** |
| `GIVES_CHECK` | 1.7559 | 3.44 per move-generating node |
| `THR_PUSH` | 1.0087 | threat plies pushed |
| `THR_SPAN_PLIES` | 0.6206 | plies whose events are generated ⇒ **61% of pushes**, fold span **1.44** |
| `THR_RAW_PUSH` / `THR_DROPPED` | 2.4208 / 0.2056 | the `threat_pair_kept` filter drops 8.5% |
| `THR_EVENTS_GEN` | 2.2152 | **3.57 events per generated ply** |
| `THR_FOLDS` / `THR_FOLD_ROWS` | 1.2005 / 4.0481 | **3.37 rows per perspective-fold** |
| `THR_FOLD_COPYONLY` | 0.3355 | **28% of threat folds fold ZERO rows** |
| `HK_FOLDS` / `HK_FOLD_ROWS` | 0.7178 / 1.6556 | 2.31 rows per fold |
| `HK_FINNY` / `HK_FINNY_ROWS` | 0.3167 / 1.5178 | **4.79 rows per king refresh**, on **36.7% of compute calls** |
| `HK_FINNY_COLD` | 0.0001 | the cold path is 128 calls in 1.5 M nodes |

Two of these overturned a belief before any code was written:

* **The threat fold is small and count-bound.** 4.05 rows/node at 120
  instructions per 1280-wide i8 row is 486 i/n, plus 1.20 copies × 80 = 96.
  The measured fold phase of `at_full` is 715 i/n. There is no algorithmic
  slack in it — which is what sent me to look at SF's fold and find parity
  (§3.2).
* **`HK_FINNY_COLD` = 128 calls total.** `avx512::i16_deltas` — the kernel
  whose ONLY production caller is that cold path — appears at **3.79% of the
  engine** in study 9's PGO flat profile. A symbol running 128 times in
  15 M nodes cannot be 3.79%: PGO had folded the Finny *dual* kernel's body
  onto that symbol. **Anyone reading 244 i/n off `apply_i16_deltas` is
  attributing it to dead code.** The check that catches this class is cheap
  and this study now recommends it by default: put a counter on the call
  before believing a flat profile's attribution.

---

## 2. WHAT LANDED

Probe harness: `scripts/probe10.sh <label>` — native non-PGO, fixed
5,001,562-node tree, prints the wl50 eval hash so a tree change cannot hide.
Raw: `probes.txt`. Every row below has hash `5ca978e0…` and node count
5,001,562, i.e. **all of it is eval bit-exact and node-identical**.

| arm | instructions | per node | Δ |
|---|---:|---:|---:|
| `base` (main) | 35,045,570,772 | 7,006.9 | — |
| **`resolve-checks`** | **34,763,399,400** | **6,950.5** | **−0.805%** |
| `finny-diff` (three changes) | 34,837,451,614 | 6,965.3 | +0.213% ⇒ **reverted** |
| **`finny-hoist-only`** | **34,744,897,614** | **6,946.8** | **−0.053%** |

### 2.1 Five panic edges out of the threat resolver (−0.805%)

`ThreatContext::compute` resolves every raw threat event to a feature index
for both perspectives — **4.43 event-perspectives per node** — and the loop
body carried **six** conditional branches into panic landing pads. Three
independent sources, all removed in safe Rust:

* **`index_decoded`'s oriented squares.** `from_o` and `to_o` are built as
  `u8 ^ u8`. Both operands are 6-bit by construction (`Square::index() < 64`;
  `orientation = orient_threats(ksq) ^ 56·p ∈ {0,7,56,63}`) but LLVM sees
  `0..=255`, so every `offsets[att][from_o]` and `lut2[att][from_o][to_o]`
  read paid a `cmp $63`/`ja` pair it could **not** hoist — the bound depends
  on the loaded event. `& 63` is a no-op on the value and a range fact for the
  optimizer. This is perf study 8 §4.1's `l1_dot` shape exactly.
* **The cursor writes.** `adds[p][len[p].0] = idx` into `[usize; DELTA_CAP]`.
  Each event pushes at most one index, so a cursor cannot exceed `DELTA_CAP`,
  but LLVM cannot relate the trip count to the array length. `DELTA_CAP` is a
  power of two, so `& (DELTA_CAP - 1)` is free; a `debug_assert` carries the
  real invariant.
* **`psqt_row`.** `table[f*BUCKETS..][..BUCKETS].try_into()` bounds-checks
  three times — the open range, the closed range, and the length.
  `as_chunks::<BUCKETS>()` states the shape once. Shared with the HalfKA fold,
  which gets the same reduction for free.

**The candidate did not come from `perf annotate` share.** It came from
`panicedge10.py`, then from reading the loop body and multiplying by
`THR_EVENTS_GEN`. Study 8's `bcheck8.py` scores this loop at zero, because
every one of these jumps targets an address *inside `at_full` itself*.

### 2.2 The Finny row-base hoist (−0.053%)

`halfka_index` recomputes `piece_square_index + bucket·PS_NB` per square. The
Finny set-diff walks one piece type's added and removed squares as two
separate bitboard loops, and paid that nine-instruction base
(`shl`/`test`/`lea`/`cmove`/`cmp`/`mov`/`cmove`/`add`) in **each** of them,
for a value constant across both. `halfka_row_base` / `halfka_oriented` split
it, with `halfka_index` delegating to the pair so the two definitions cannot
drift — the same construction as `ThreatResolver::index` / `index_decoded`.
The cursor writes take the same mask as §2.1.

Small, and quoted as small. It is kept because the probe pair is exact
(identical source and flags ⇒ identical binary), it is bit-exact, and the
decomposition is the right shape.

---

## 3. KILLED BY A NUMBER

### 3.1 Iterating bitboard rows instead of `PieceType::ALL` (+0.266%)

The Finny diff loop's disassembly shows LLVM materialising the 6-byte
`PieceType::ALL` array **on the stack every call** (`movw $0x504,0xbc(%rsp)`,
`movl $0x3020100,0xb8(%rsp)`) and reloading the piece type from it each
iteration (`movzbl 0xb8(%rsp,%r13,1),%r12d`). Zipping the two `[Bitboard; 6]`
rows instead gives a compile-time trip count and no memory for the loop
variable.

It measured **+0.213%** batched with §2.2, i.e. **+0.266% on its own** — it
costs more than four times what §2.2 saves. The mechanism is study 8 §7.9's:
`old_row.iter_mut()` holds a mutable borrow of the cache row across the whole
walk, and the aliasing facts that costs are worth more than the stack traffic
it removes. **Recorded so it is not re-litigated: on this codebase, replacing
an indexed loop with a zipped-iterator loop over a `&mut` row is a
pessimisation, twice now.**

### 3.2 "Make the threat channel incremental" — the premise is false

The mandate named this "the single largest structural item": *`at_full` exists
because our threats are replayed, not maintained. SF has no equivalent stage
at all.*

**SF has exactly the same stage, and pays the same for it.**

`vpmovsxbw` is the i8→i16 widening. In both engines the FullThreats
transformer is the only i8-weight table (`ThreatWeightType = std::int8_t`,
`nnue_common.h:52`; ours `THREAT_QUANT = 64`), so **`vpmovsxbw` executes if
and only if a threat accumulator row is being folded.** It is a signature
opcode, and it needs no source reading and no symbol attribution.

Instruction-attributed annotate, both engines, same panel
(`scripts/sfsplit10.sh`; raw `annot-sf-evaluate.txt`, `annot-at_full2.txt`):

| | `vpmovsxbw`, per node | where |
|---|---:|---|
| **SF18** | **219.7 i/n** | inside `Network<1024>::evaluate` |
| **ARCHON** | **216.5 i/n** | inside `ThreatContext::at_full` |

Within 1.5%. And since SF's accumulator is 1024 lanes (32 `vpmovsxbw` per
row) against our 1280 (40 per row), **SF folds ~6.9 threat rows per node to
our ~5.4** — it folds *more*, not fewer.

SF's structure is also ours, not a different design: `DirtyThreatList` of ≤96
events appended during `do_move` (`update_piece_threats`), resolved per
perspective by `append_changed_indices` into add/remove index lists, then
applied by a register-tiled loop that holds a tile of the accumulator across
all rows (`nnue_accumulator.cpp`). Our `apply_i8_deltas_into` is the same
8-register tiled shape.

**What is different is the scalar half.** Splitting both symbols by opcode
class:

| | SIMD share | scalar share |
|---|---:|---:|
| SF18 `Network<1024>::evaluate` | 70% | 30% |
| ARCHON `at_full` | 55% | 45% |

Splitting our threat channel (≈1,498 i/n in the PGO frame) the same way:
`at_full`'s SIMD is `vpmovsxbw` 216.5 + `vpaddw` 185.3 + `vpsubw` 70.5 +
`vmovdqu64` 112.3 = **585 i/n**, plus `refresh_threat` 60 ⇒ **≈645 i/n of
fold, ≈853 i/n of scalar**. On SF's side the fold is 219.7 of widening plus
its paired adds and loads (≈440), and the scalar is `update_piece_threats`
230 + `swap_piece` 59 plus the index resolution inside `evaluate` (part of
its 332 i/n of scalar) ⇒ **≈400 i/n**.

**Our threat fold is at parity; our threat SCALAR is ≈2.1× SF's, ≈450 i/n of
excess** — event generation, per-perspective resolution, and the per-push
position snapshot. §2.1 took 56 i/n out of exactly that, which is the first
evidence that the plumbing is where the remaining threat money is.

### 3.3 `set_check_info` laziness — the read rate, measured

Study 9 priced it at 106 i/n (1.50%) and declined it on **an argument**: the
state is read per-move by `gives_check`, so a laziness check might cost more
than it saves. The mandate asked for the read rate. Here it is:

| | per node |
|---|---:|
| `set_check_info` fills (`make_move` + `null_move`) | **1.0125** |
| nodes that generate moves (`MovePicker::new` + `new_probcut`) | **0.5103** |
| `gives_check` calls | 1.7559 |

**49.6% of the fills are thrown away.** The reads are *concentrated*, not
spread: a node either generates moves and then reads the state 3.44 times, or
never touches it. So the right design is a single node-granular `ensure`, not
a test per read — which makes the ensure cost ~2 i/n, not ~15.

Ceiling: **0.496 × 78 i/n = 39 i/n = 0.56%** (78 i/n is this study's non-PGO
measurement of the symbol; study 9's 106 i/n is the PGO-frame idempotent
double-call, and 0.496 × 106 = 53 i/n in that frame).

**Declined, on the number.** Making the fill lazy means removing it from
`make_move_with` and threading an `ensure` through `MovePicker::new`,
`new_probcut`, root move generation, perft, `threads.rs` and the UCI paths —
a movegen-contract change whose every missed call site is a silent wrong
answer, for 0.56% of instructions ⇒ ≈ +0.25% effective speed ⇒ **≈ +0.4 Elo**
at the measured conversion rate. The disposition the mandate asked for is
now on the number rather than on the argument, and the number does not
justify the regression surface. If it is revisited, the safe construction is
a `#[cfg(debug_assertions)]` validity flag in `StateInfo` so the whole test
suite (which runs in debug) panics on any missed site.

### 3.4 `init_quiets`' panic edges — a candidate list, not a ranking

`panicedge10.py` finds 57 static panic edges in `init_quiets`, and weighting
them by annotate samples puts **13.71% of the symbol** on them — 70 i/n, the
largest weighted block the tool found anywhere. It was **not attempted**,
because study 8 §7.1 already implemented this exact class and measured
**−0.045%**: the weight is store/compare skid, which this box produces
reliably on that symbol. The tool is a candidate list; the ranking has to come
from a count or a stub, and the count here (the insertion sort's trip count)
does not support 70 i/n.

---

## 4. THE TWO ARCHITECTURAL ITEMS, PRICED

Neither is engine surgery, and one of them is not ours to fix.

### 4.1 SF18 has adaptive eval cost and we do not — 325–740 i/n of the target

```
sf_18:src/evaluate.cpp:49   use_smallnet(pos) { return abs(simple_eval(pos)) > 962; }
sf_18:src/evaluate.cpp:62   smallNet ? networks.small.evaluate(...) : networks.big.evaluate(...)
sf_18:src/evaluate.cpp:68   if (smallNet && abs(nnue) < 277) -> redo with the big net
```

SF18 runs a **128-wide** net on lopsided positions and pays for its 1024-wide
net only near equality. **Its 4,329.6 i/n is a blend of a cheap path and an
expensive one. We run the full 1280-wide net on every node.**

`simple_eval` is exactly our `material_imbalance` (pawn-count delta × PawnValue
plus non-pawn material), and **PawnValue = 208 on both sides**
(`sf_18:src/types.h:185`, `src/types.rs:636`), so the 962 threshold transfers
with no frame conversion: **4.63 pawns.**

**How much of SF's budget is the cheap path.** From SF's own flat profile, with
`r` = the per-eval cost of the 128 net as a fraction of the 1024 net's, and `f`
= the fraction of evaluated nodes taking it:

* big-net group (`Network<1024>::evaluate` + its four accumulator helpers) = **33.25%**
* small-net group (`Network<128>::evaluate` + its three helpers) = **2.41%**
* ⇒ `f = (0.0725/r) / (1 + 0.0725/r)`

| `r` | 0.125 (pure width) | 0.15 | 0.20 | 0.25 |
|---|---:|---:|---:|---:|
| **`f`** | **36.7%** | **32.6%** | **26.6%** | **22.5%** |

**SF18 with its big net on every evaluated node would be ~4,650–5,070 i/n,
7–17% above the 4,329.6 the 1.10× bar is set from.** Restated as the mandate
frames it:

> **Between 325 and 740 of the 1,667 i/n I was asked to remove — 20% to 44%,
> centre ≈31% — is SF18's adaptive-eval feature, not our inefficiency.
> Optimising our expensive path cannot close it.**

[D, derived from a profile with a swept assumption; the assumption is `r`, and
the direct measurement that would replace it is an instrumented SF18, which
this study did not build.]

Worth recording alongside: `eca43a97` — the revision our constants are
transcribed from, 223 commits *newer* than the `sf_18` tag — is reported to
have **deleted** the small net, i.e. SF itself later judged the complexity not
worth it. **NOT VERIFIED HERE**: this worktree's isolation blocks git access to
the reference clone, and the checkout on disk is `sf_18`, which still has it.
It should be checked before being used in an argument.

### 4.2 The same gate for ARCHON — priced, and the substitute does not exist

Measured directly, at node level, over 647,556 evaluated nodes
(`perfcount`, `MAT_GT_*`):

| threshold | pawns | share of evaluated nodes |
|---:|---:|---:|
| 400 | 1.9 | 44.9% |
| 600 | 2.9 | 31.8% |
| 800 | 3.8 | 27.2% |
| **962 (SF18's own)** | **4.6** | **21.6%** |
| 1200 | 5.8 | 15.7% |
| 1600 | 7.7 | 8.6% |

**Ceiling.** Our head forward is 726 i/n over 0.4315 evals/node = 1,683
instructions per eval; the whole eval layer is 3,388 i/n = 7,852 per eval.
At 21.6%:

* **floor of the ceiling — head forward only: 157 i/n (2.4%)**
* absolute ceiling — the entire eval layer: 732 i/n (11.4%)

**The truth is near the low end**, and study 9 §3.1 is why: a lazily-folded
accumulator stack does not save work by skipping a ply, because the next fold
walks a longer span and does it anyway. Only folds at plies whose subtree ends
there are genuinely deleted.

**And the substitute net does not exist.** The gate machinery is already in
`SotaContext` (`ARCHON_SMALLNET`, `ARCHON_SMALLNET_THRESH`,
`ARCHON_SMALLNET_MAT_GATE`, perf study 3 item 8) and is OFF by default for a
documented reason: our cheap net is HalfKA-only-**768**, ~0.6× the big net, so
gating on it is a wash. The two available substitutes bracket the problem:

| substitute | cost ratio | accuracy |
|---|---:|---|
| PSQT (exists, default build) | ~0.05× | a piece-square table against a 1280-wide NNUE — a large, unpriced fidelity loss |
| HalfKA-only-768 (exists) | ~0.6× | small loss, but only 0.4× saving ⇒ the documented wash |
| **128–256-wide net (SF's shape)** | **~0.15×** | small loss — **needs a retrain** |

So this is a **legitimate finding with a stated dependency**: worth
157–732 i/n, mechanism identical to the engine we are measured against,
machinery already present, blocked on a net nobody has trained. The fidelity
half is **unpriced** — the next step is `evaldiff.py` over the 800-position
battery restricted to `|material_imbalance| > 962`, and SF's own `< 277`
recheck is a strong hint about the guard shape that would be needed.

---

## 5. DISTANCE TO 1.10×, WITH THE PER-ITEM ACCOUNTING

**Target 4,762.6 i/n. Baseline 6,429.9. Required: −1,667.3 i/n. Landed:
−62.8. Remaining: −1,604.5.**

Block decomposition, ARCHON against SF18, with the eval budget grossed up by
study 8 §1's measured **1.080× width premium** applied to the eval block only:

| block | ARCHON i/n | SF18 i/n | width-justified budget | excess |
|---|---:|---:|---:|---:|
| eval layer [±6%, see §6.1] | 3,388 | 1,839 | 2,188 | **+1,199** |
| search shell | 2,276 | 1,958 | 1,958 | +318 |
| make / unmake | 591 | 276 | 276 | +315 |
| startup (net load, magics) | 176 | 181 | 181 | −5 |
| **total** | **6,430** | **4,254** | **4,603** | **+1,827** |

(SF18's blocks sum to 4,254 against its measured 4,329.6; the 76 i/n
difference is symbols below the 0.05% profile cut, and it sits in SF's favour
— counting it would make the excess larger, not smaller.)

Where the 1,667 would have to come from, and what each candidate measured:

| source | i/n | status |
|---|---:|---|
| **SF18's small-net blend — not ours to remove** | **325–740** | §4.1, priced, structural |
| An ARCHON eval gate (needs a 128–256-wide net) | 157–732 | §4.2, priced, **retrain-dependent**, fidelity unpriced |
| Threat scalar plumbing over SF's | ~460 | §3.2; §2.1 took **56**, the rest is unpriced |
| Search shell over SF's (1.16×) | 318 | never opened; movepick already at parity (study 8 §7.6) |
| make/unmake over SF's (2.1×) | 315 | of which `set_check_info` laziness = **39**, measured and declined (§3.3) |
| HalfKA Finny refresh scalar half (6.8× SF's: 184 vs 27 i/n) | ~157 | §3.1 attacked it and **lost**; §2.2 took 3.7 |
| Threat FOLD | **0** | **at parity** — 216.5 vs 219.7 i/n (§3.2) |
| `l2_tail` + `l3` | **0** | at parity, 1.03× (study 9 §3.5) |
| SMP scaling | **0** | at parity, 10.45× vs 10.47× (study 9 §4.2) |
| Accumulator copies | ~30 | the remainder are memoizations (study 8 §7.10, +1.04% to remove) |
| **landed by this study** | **−62.8** | §2 |

**The honest answer.** Deleting the **entire** FullThreats channel — a channel
SF18 also has — is worth 23.3% of the engine (1,617 i/n by this study's own
symbol accounting; the NNUE study's independent cost model says 23.8%, and the
two agree to 0.7%). That lands at **4,932 i/n = 1.139× SF18** — **still above
the 1.10× bar**, while giving up the eval quality the channel buys and needing
a retrain. Reaching 1.10× additionally requires narrowing the accumulator to
1024, at which point the arithmetic works (≈4,650 i/n, 1.07×) but the engine
is a strictly weaker architecture than both D48X and SF18.

**So the bar is not reachable by implementation, and it is not reachable by
subtraction either.** ~31% of it is a feature SF18 has and we do not; the
largest single block the mandate pointed at is at parity; and the rest is
distributed across three subsystems in units of 300 i/n with no single lever
above ~2.5% anywhere on the board. The three most recent studies have removed
**749.5 i/n** between them across **twelve** landed changes (study 8: 570.6 in
six; study 9: 116.1 in three; study 10: 62.8 in two), largest single change
2.43%. **1,604.5 i/n remain.**

**What the target is worth, at the measured conversion rate.** −1,667 i/n is
−26%; at 0.45× conversion that is ≈ 0.18 doublings ⇒ **+21 Elo at 120
Elo/doubling, +12 at 70.** Worth having, and worth knowing before spending
another study on items worth <1% each.

### 5.1 What I would do next, in order

1. **Price the eval gate properly (§4.2)** — it is the only uncosted item left
   above 150 i/n. The missing half is fidelity, and it costs one `evaldiff.py`
   run restricted to `|material_imbalance| > 962`, not a training run. If the
   fidelity is acceptable with a PSQT substitute, the ceiling is available with
   no retrain at all.
2. **The threat channel's scalar plumbing** — 460 i/n, now the largest item
   that is genuinely ours. §2.1 proved there is money in it. `at_full`'s
   resolve phase is still ~290 i/n for 2.2 events/node.
3. **Not the fold, not the tail, not SMP, not movepick, not the threat
   systems' unification, not `set_check_info`.** Six directions now closed
   with numbers.

---

## 6. UNCERTAINTIES AND THREATS TO VALIDITY

1. **The block table in §5 mixes frames.** ARCHON's block shares come from a
   PGO flat profile taken at study 8's HEAD (6,546 i/n) and are scaled to the
   current 6,430; SF18's come from this session's own profile. The same blocks
   read off this study's own **non-PGO** profile put the eval layer at 46.2%
   rather than 52.7% — 3,211 vs 3,388 i/n — because non-PGO leaves small
   callees outlined and PGO inlines them into `search`. **Quote the eval block
   as 3,200–3,400 i/n**; the sign and the ordering of §5's rows are unaffected,
   but no single-symbol split inside it should be quoted to better than ±6%.
2. **Non-PGO profiles overstate small functions.** `correction_value` is 2.40%
   of the non-PGO profile and absent from the PGO one (inlined into `search`).
   Every candidate here was priced on a probe *pair*, where that bias cancels,
   but the absolute i/n attributions in §5 carry it.
3. **§4.1's `f` is derived, not measured.** It rests on `r`, the small net's
   per-eval cost ratio, swept 0.125–0.25. The range 23–37% is the sweep, not a
   confidence interval.
4. **§4.2's ceiling is a range spanning 4.7×** because the accumulator-fold
   saving depends on how many gated nodes are subtree leaves, which was not
   measured. Only the 157 i/n head-forward floor is solid.
5. **`vpmovsxbw` attribution assumes it appears nowhere else in SF's eval.**
   That follows from `ThreatWeightType = int8_t` being the only i8 weight table
   in the transformer, but it is a source reading, not an instrumented count.
   The claim is robust to the *direction* it would move if wrong (any other
   consumer would make SF's threat fold look even cheaper than ours, which is
   the opposite of the mandate's premise).
6. **Nothing here was Elo-tested**, and §1's arithmetic says nothing here
   *could* be: −0.977% is ≈ +0.44% effective speed, i.e. ≈ 0.013 ply.
7. **Box state.** Load 5–10 and 1–3 GB available for the whole session (another
   agent's multi-engine gauntlet), swap full at times. Every claim rides on
   `instructions:u` on fixed-node trees, which is contention-immune and
   reproduced at 0.0034% spread. **No cycles or NPS number in this document is
   offered as a claim.**

---

## 7. ARTIFACTS (`perf-artifacts-10/`)

* `counters-base.txt` / `counters-final.txt` — the matched fixed-node panels.
* `counters-perfcount.txt` — the §1.1 rate table.
* `probes.txt` — every arm, with its eval hash and node count.
* `counters.patch` — the `perfcount` instrumentation, reverted after use.
* `flat-{s10base-native,s10a,sf18-s10}-instructions.txt` — flat profiles.
* `annot-*.txt` — per-symbol disassembly with instruction-event weights,
  including `annot-sf-evaluate.txt` (§3.2's opcode split) and
  `annot-sf-refresh.txt`.
* `scripts/`
  * `panicedge10.py` — **new**: panic edges by enclosing symbol, seeing the
    landing pads LLVM sinks into hot functions (which `bcheck8.py` cannot).
  * `sfsplit10.sh` — **new**: annotate SF18's big-net `evaluate` for the
    opcode-class split that makes the SF-side accounting possible.
  * `probe10.sh` — the non-PGO pricing pair, retargeted at this study.
  * everything else inherited from `perf-artifacts-9/scripts/`.

**Reproduction.**

```bash
# identity pins (unchanged by both landed changes)
./engines/bin/archon-s10-final nnue-eval engines/canonical/net.cenn \
  < tools/nnue/calib/wl50.fens | sha256sum        # -> 5ca978e0…8b38
taskset -c 4 ./engines/bin/archon-s10-final bench 16 1 13 \
  tools/nnue/calib/wl50.fens depth engines/canonical/net.cenn   # -> 1,922,947

# the matched panel
bash docs/research/perf-artifacts-10/scripts/pstat8.sh 6 300000 \
  tools/nnue/calib/wl50.fens /tmp/counters.txt \
  "final:./engines/bin/archon-s10-final @BENCH@ engines/canonical/net.cenn" \
  "sf18:./engines/bin/stockfish18 @BENCH@"
python3 docs/research/perf-artifacts-10/scripts/parse8.py /tmp/counters.txt sf18

# the panic-edge candidate list
python3 docs/research/perf-artifacts-10/scripts/panicedge10.py \
  target-probe10/release/chessengine 25

# SF18's threat-fold share, by opcode
bash docs/research/perf-artifacts-10/scripts/sfsplit10.sh \
  docs/research/perf-artifacts-10/perf-sf18-s10-instructions.data /tmp/sf.txt
```
