# ARCHON perf study #31 — the road to Stockfish-class per-node speed: a costed, ranked plan

Date: 2026-07-30. Raw artifacts and instruments: `docs/research/perf-artifacts-31/`,
`tools/perf31/`. Landed this session: one arm (`4503060`), which exists to
*refute* a mechanism rather than to buy speed.

**This is a ROADMAP, not an optimisation sweep.** Thirteen studies have shaved
instructions; this one asks what the remaining 1.37× is made of, what it would
take to remove each part, what that is worth in Elo, and what should not be
attempted. Every number below is labelled **measured**, **derived**,
**bracketed** or **estimated**, and every instrument is quoted with its own
null.

---

## 0. HEADLINE

### 0.1 Five claims, in the order they change the plan

1. **52% of our load-stall cycles are feature-transformer weight rows, and 61%
   of those are the HalfKA table, not the threat table.** Nobody has attacked
   either. This is the largest identified pool in the engine and it is
   §3's central result.
2. **Accumulator streaming — the thing "fuse the two accumulators" attacks — is
   19% of load-stall cycles, and I measured its price directly.** One extra
   accumulator-pair copy per evaluated node costs **+32.88 L1 misses/node and
   +162.7 load-stall cycles/node** against nulls of −0.13% and −2.7% (§2.4).
3. **Fusion is therefore worth about −136 cycles/node (≈ +4.7 Elo), not
   −150…−375 and not the +872 i/n loss that closed it.** Both prior verdicts
   are wrong, for a reason I can name: the refresh cost that killed it was
   priced at 6,240 instructions per rebuild, and the rebuild is **4.18 rows**
   (§2.3, measured tonight). Study 28 ranked fusion #1; it is #4.
4. **Study 28's item C — the per-node `ThreatPosition` snapshot, priced at
   −264 cycles/node — is a NULL.** I built the arm and measured it: the
   L1-miss counter, whose null here is ±0.16%, does not move (§2.2). The
   +303.4 cycles/node `search::<NonPv>` stalls above Stockfish is a *site*
   with no mechanism.
5. **The realised-versus-predicted record says how to read all of this, and it
   depends on how an item was priced.** Across studies 8–19, items priced as
   *"this edit removes N instructions at M sites"* realised **76.5%**; items
   priced as *"we are N i/n worse than Stockfish at this block"* realised
   **6.8%** (§5). Study 28's four ceilings are all block-excess prices.

### 0.2 The plan in one table

Cycles/node off the shipped PGO baseline **4106.8** (Stockfish `eca43a97`:
**3005.0**). Elo at the project's own constant, **120 Elo per doubling of
speed** (exp-0154: 1.51× ≈ +73 Elo ⇒ 122.9/doubling).

⚠ **Frame.** Items 1, 2 and 4 are priced in §1.1's non-PGO probe frame
(base 5068.6 cycles/node); items 3, 5 and 6 are inherited PGO-frame numbers
(base 4106.8). Elo is computed against each item's own base, which keeps the
*ratios* comparable even though the absolute cycle counts are not. Full
per-node parity with Stockfish is **+54.1 Elo**.

| # | item | cycles/node | evidence | Elo if realised | blocked on |
|---|---|---:|---|---:|---|
| **1** | **HalfKA weight-row bytes** (57.7 MB table, 125.2 lines/node) | **−926 ceiling** | derived, §3.3 | +35 | a measurement, then a retrain |
| **2** | **Threat weight-row bytes** (77.7 MB table, 79.9 lines/node) | **−591 ceiling** | derived, §3.3 | +21 | a co-access study |
| **3** | **`search::<NonPv>`'s +303.4** | **−303 ceiling, mechanism UNKNOWN** | site only, §4.2 | +13 | one profiling session |
| **4** | fuse the two accumulators | **−136** [−183, −89] | derived from a measured probe, §2.4 | +4.7 | nothing — it is buildable now |
| **5** | width 1280 → 1024 | **−271** | measured intervention (study 28 §4) | +11.8 **but Elo-neutral**, §6.2 | a retrain |
| **6** | front-end + retiring + other-exec excess | −196 ceiling | measured (study 28 §0.3) | +8.5 | — |
| — | *per-node snapshot locality* | **0, MEASURED** | §2.2 | 0 | closed |
| — | *further instruction micro-optimisation* | ~0 | §6.1 | 0 | closed |

Items 1 and 2 are the same mechanism at two tables and they are **not
independent of item 5**: width is a 20% cut of both. Items 1–3 have ceilings
and no mechanism yet; item 4 has a mechanism and a small number. That is the
honest shape of the thing.

### 0.3 What I would actually do, and what it is worth

**Three measurements, then one build.** In order:

1. **Localise `search::<NonPv>`'s +303.4 cycles/node** (§4.2). One
   `perf record -e ex_no_retire.load_not_complete` with the accumulator
   `push`, the TT probe and the history planes each behind `#[inline(never)]`.
   ~2 hours. It is 26.7% of the gap and its only named mechanism is now dead.
2. **A co-access census of the two FT tables** (§4.1). Which feature indices
   fire together within one fold, and within one refresh? One instrumented
   build, one histogram. It prices items 1 and 2 and it is the input to the
   only lever that can attack them without a retrain.
3. **The i8 HalfKA row** (§4.3). Halving the HalfKA row from 2560 B to 1280 B
   is a **−463 cycles/node** structural change — larger than fusion and width
   combined — and it is a quantisation question, answerable by the existing
   800-FEN fidelity battery before any engine work.

Then build fusion (item 4), because it is the only item on the list whose
mechanism, price and blocker are all known.

**Expected value, discounted by this project's own record (§5):** **+8 to +18
Elo over a serious multi-week effort**, landing at 1.15–1.25× Stockfish per
node. Study 28 said +22 to +32; I am below it because its item B is smaller
than it thought, its item C is zero, and its items A and D are priced in the
class that historically realises 6.8%. **The upside above that band lives
entirely in items 1–3, which are unpriced — and that is an argument for
measuring them, not for assuming they are empty.**

---

## 1. THE INSTRUMENTS, AND WHAT EACH NUMBER HERE CAN SUPPORT

This section is not boilerplate. Two studies retracted committed cycles
headlines on this box, and a roadmap about *cycles* has to say up front which
of its numbers are real.

### 1.1 My own frame

`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
— 50 positions × 100 k nodes = **5,001,562 nodes**, eval hash
`5ca978e012da1bb0`, `--features sota,arch-d48x`, D48X net, **fat LTO,
CGU = 1** (the shipped release profile — study 26 §0 showed that at 4 CGUs even
`instructions:u` is unreliable), `target-cpu=native`, **non-PGO**,
`taskset -c 6`, `nice -n 19`.

Baseline: **5653.84 i/n, 588.65 br/n**. The shipped PGO engine is 5362.4 i/n;
this is the probe frame and its deltas historically over- *or* under-state PGO
by 1.1–2.9× in both directions (study 19 §7).

**The box was contended throughout** — the D48X trainer plus two sibling perf
studies plus a browser, load average ~5 on 8 cores. This inflates the cycles
and stall nulls below and is stated with every one of them.

### 1.2 The nulls I measured, tonight, interleaved with the arms

`tools/perf31/stalls31.sh` runs a three-way round-robin — base, a
**byte-identical copy** of the base, and the arm — with the **slot order
randomised per round** (study 25 §0.2 measured that a *fixed*-order round-robin
cancels monotone drift but not curvature: its fixed-order null read +2.06% and
its randomised one −1.93%). 12 rounds each.

| counter | null, panel 1 | null, panel 2 | admissible claim |
|---|---:|---:|---|
| `instructions:u` | **+0.000%** | **−0.000%** | anything above ~0.02% |
| `L1-dcache-loads:u` | +0.005% | +0.116% | above ~0.3% |
| `L1-dcache-load-misses:u` | **−0.160%** | **−0.125%** | **above ~0.5%** |
| `ex_no_retire.load_not_complete:u` | +2.100% | −2.709% | above ~6% |
| `cycles:u` | +2.067% | −2.876% | above ~7% |

**The L1-miss counter is the useful one and it is much better than its
reputation.** At ±0.16% on 618 misses/node it resolves **±1 miss per node**.
Every load-bearing measurement below is anchored on it; the stall counter is
quoted where the effect clears ±6% and never otherwise; **no cycles claim in
this document rests on `cycles:u`.**

### 1.3 Inherited, not re-derived (study 28, PGO, quiet box)

| | ARCHON | Stockfish `eca43a97` | ratio |
|---|---:|---:|---:|
| cycles/node | 4106.8 | 3005.0 | **1.3667×** |
| instructions/node | 5362.4 | 4383.0 | 1.2235× |
| IPC | 1.306 | 1.459 | 1.117× |
| L1d loads/node | 2947.6 | 2229.3 | 1.322× |
| L1d load-misses/node | 588.1 | 438.7 | 1.340× |
| L1 miss **rate** | 19.95% | 19.68% | 1.014× |
| load-stall cycles/node | 2458.6 | 1517.5 | 1.620× |
| evals/node | 0.4307 | 0.4327 | 0.995× |

82.8% of the +1136.9 cycle excess is `ex_no_retire.load_not_complete`. The gap
is **1.37× quiet, 1.48× loaded** — asymmetric because we are 59% load-stalled
against their 50%, so L3 contention lengthens our stalls more.

**Also inherited and load-bearing:** Reckless (Rust, top-2, ~3833 SPCC) costs
**4084.4 cycles/node against our 4106.8**. Two independently developed engines
land 0.5% apart. Neither the language nor our code quality is the explanation,
and I have not re-tested either.

---

## 2. THREE MEASUREMENTS I MADE, AND WHAT THEY DECIDE

### 2.1 A per-node census of the NNUE feature transformer — and its own control

A throwaway instrumented build (`bin31/s31-ctr`, ten `Relaxed` counters,
reverted before the commit) on the frame above:

```
thr_refresh/n      = 0.01955   thr_refresh_rows/call = 4.18    thr_refresh_rows/n = 0.082
thr_fold/n         = 0.8498    thr_add/fold = 1.970   thr_sub/fold = 2.633   thr_rows/n = 3.912
hka_fold/n         = 0.9749    hka_rows/fold = 2.275                          hka_rows/n = 2.218
hka_refresh/n      = 0.16102                          hka_refresh_rows/n = 0.9132
push/n             = 1.0000
```

**The instrument's control is that four of these were already known and it
reproduces all four:** `hka_refresh/n` 0.16102 against study 12's 0.1611;
`hka_refresh_rows/n` 0.9132 against 0.9064; `hka_rows/fold` 2.275 against study
25's dispatch-weighted 2.27; and `push/n` exactly 1.0000, which it must be
because the search pushes once per node.

**The number nobody had:** a from-scratch FullThreats rebuild adds **4.18
rows**, not the dozens everyone has assumed. (Stockfish's own cap is 128 —
`full_threats.h:67`.) On these 50 calibration positions, many of them
endgames, the kept-victim masks leave very few live threats. This single
number is what re-opens fusion in §2.3.

### 2.2 Arm p1 — study 28's item C is a null, and the mechanism is excluded

**The hypothesis, quoting study 28 §7 and §11.1 verbatim:** `search::<NonPv>`
stalls **3.03× Stockfish's search symbol**, +303.4 cycles/node, **26.7% of the
whole gap**, because "ARCHON's search node inlines the accumulator `push`: a
`DirtyPiece` and an **80-byte `ThreatPosition` snapshot** written every node
into a stack whose per-ply stride is ~10.5 KB … a write into a cold line at
every single node, evaluated or not". Priced at **−264 cycles/node** and
flagged for re-measurement "on cycles and load-stalls" because the only prior
rejection (`sota_threats.rs:1032-1048`, +106 i/n) was an instruction-count
rejection.

**The arm:** move `dp` + `position` out of the 5,312-byte `ThreatEntry` into
their own dense `Box<[Push; STACK_LEN]>`. The every-node write goes from a
**5,312-byte stride** — two guaranteed-distinct, unprefetchable lines — to an
**88-byte** one, where consecutive plies share a line and the writes are a
sequential stream. This is the cheapest possible test of the *locality* claim
and it is node-identical and eval-bit-identical by construction.

| counter | base | NULL | ARM |
|---|---:|---:|---:|
| L1-dcache-load-misses/node | 618.08 | −0.160% | **+0.254% (+1.57)** |
| `ex_no_retire.load_not_complete`/node | 2836.01 | +2.100% | +3.583% |
| cycles/node | 4910.26 | +2.067% | +2.369% |
| instructions/node | 5653.84 | +0.000% | −0.014% (−0.78) |
| L1-dcache-loads/node | 3043.59 | +0.005% | +0.023% |

**Nothing moves.** On the counter that resolves ±1 miss/node, converting every
node's snapshot write from a 5.3 KB stride to an 88-byte one is worth
**+1.57 misses/node** — the wrong sign, inside the null. A −264 cycles/node
mechanism would have to be worth roughly 50 misses/node.

**Why, in hindsight:** only ~15 plies are live at a time and
`ThreatContext::compute` walks them repeatedly on every evaluated node, so the
top of a 5.3 KB-strided stack is L1/L2-resident. It was never a cold line.

**What this does to item C:** the +303.4 cycles/node is real (study 28
measured it with `perf record` on the stall event itself) but it is a **site**,
not a mechanism, and its only named mechanism is now excluded. It stays on the
roadmap at #3 as *a measurement*, not as an optimisation.

The arm is landed anyway (`4503060`) at −0.78 i/n / −0.44 br/n against an exact
0.000% null, 12/12 sign-consistent, with the refutation written into the
`Push` doc comment so the split is not proposed a third time.

### 2.3 The fusion blocker, re-priced — and Stockfish read correctly

Three studies have reached three different verdicts on fusing the HalfKA and
FullThreats accumulators into one array. They can now be reconciled.

**What Stockfish actually does at `eca43a97`** (verified by me directly against
the commit object, not the checkout — `sf_18` is 223 commits *older* and has
two stacks, which is the error study 22 made):

* One `i16[1024]` accumulator per side per ply. Threat rows are `i8`, widened
  and added **in place** into the same array (`nnue_accumulator.cpp`,
  `apply_combined`).
* **The Finny cache holds the HalfKA contribution only.** The comment above
  `update_accumulator_refresh_cache` says so: *"HalfKA data comes from the
  Finny table entry, while the threats are built from the active threat
  features"*, and the function calls
  `ThreatFeatureSet::append_active_indices(perspective, pos, active)` — the
  **full** active set, from scratch, on **every** refresh.
* `find_last_usable_accumulator` tests exactly one predicate, and its comment
  is the whole story: *"Threat feature set refreshes require a king move
  across the center, i.e., **a subset of halfka refreshes**"*.

**So Stockfish did not solve the blocker. It pays it.** Because its
accumulator is fused, it cannot exploit the fact that threats only need
re-orienting on a d/e mirror crossing, and it rebuilds the whole active threat
set on every own-king move (0.1673/node) where we rebuild on 0.0214. Our split
accumulator is a **genuine structural advantage over Stockfish on this one
axis**, and fusion trades it away. `sota_threats.rs:81-93` already said this
and cited the exact line; it was right.

**The arithmetic that closed fusion, and why it is wrong.** Study 22 §3:

```
cost = (0.1611 − 0.0214) × 6,240 = +872 i/n     against a measured −157 i/n saving
```

The rates are right. **The 6,240 is not.** It came from study 17 §3.3 —
222.7 i/n at 0.0357 calls/node, callgrind, for the *eager* refresh **stage** of
`do_move`. Study 18 §5.2 then corrected its own predecessor twice: the
`refresh_threat_into` symbol is **2,184 i/call**, and the 2,560-byte
`dst.fill(0)` beside it is glibc `rep stosb`, which **callgrind counts once per
byte** (2,237 Ir) while `perf` retires it as ~one instruction. Study 19's
matched 30 k-node panel puts the whole refresh column at **57.28 i/n at 0.0204
calls = 2,808 i/call**.

And tonight's census says why it is even smaller than that in the perf frame:
**the rebuild adds 4.18 rows.** At ~180 retired instructions per 1280-lane i8
row plus a board sweep, a rebuild is **~1,300 retired instructions**, not
6,240. Study 22 hedged that "the cost would have to be over-stated by **8×**
before the merge broke even"; it is over-stated by **2.2–4.8×**, and the
saving side was under-counted (it omitted `activate`'s share, ~69 i/n by study
12's count).

| fusion, instructions/node | value | source |
|---|---:|---|
| saving: one fold pass's copy | −133 to −157 | **measured** (§2.4; study 22 §3) |
| saving: `activate` loses 6 of 16 ops/iteration | −69 | derived (study 12 §3) |
| cost: 0.14147 extra rebuilds × ~1,300 | +184 | derived from tonight's 4.18 rows |
| cost, callgrind-framed upper bound (× 2,808) | +397 | study 19 §4.1 |
| **net** | **−42 … +199** | |

Instruction-neutral, then — which is exactly the point, because §1.3 says
instructions are not the binding constraint.

### 2.4 What accumulator streaming actually costs — the probe that prices fusion

The saving side needed a number in the right currency, so I measured it
directly: **one redundant `[[i16; ACC]; 2]` copy per evaluated node**
(5,120 B read + 5,120 B written = 160 cache lines per evaluated node =
**68.9 lines/node**), `black_box`'d so it cannot be eliminated. This is
exactly the traffic a fused accumulator deletes.

| counter | base | NULL | PROBE |
|---|---:|---:|---:|
| **L1-dcache-load-misses/node** | 620.73 | −0.125% | **+5.296% (+32.88)**, 0/12 negative |
| **`ex_no_retire.load_not_complete`/node** | 2934.35 | −2.709% | **+5.905% (+162.69)** |
| L1-dcache-loads/node | 3035.40 | +0.116% | +5.075% (+153.86), 0/12 |
| instructions/node | 5653.06 | −0.000% | +2.360% (+133.41), 0/12 |
| cycles/node | 5068.62 | −2.876% | +3.854% (+184.18) |

**The L1-miss row clears its null by 42×** and is sign-consistent in all 12
rounds. The stall row clears its null by 2.2×. So:

> **68.9 lines of accumulator streaming = 32.88 L1 misses + 162.7 load-stall
> cycles + 133.4 instructions per node.**
> ⇒ **0.477 misses per line, 2.36 stall cycles per line, 4.95 stall cycles per
> miss.**

That last figure matters: accumulator misses are **not** cheap. At 4.95 cycles
each they cost the same as the engine's average miss (study 28: 2458.6/588.1 =
4.18). The intuition that a recently-written stack is warm and therefore free
is wrong — 48% of those lines miss L1.

**Fusion removes 102.5 lines/node** of accumulator streaming: one of the two
parent→child fold passes (0.8498 folds/node × 80 lines = 68.0) plus half of
`activate`'s read (0.4307 evals × 80 = 34.5). That is **1.488 probe-equivalents**:

| fusion, cycles/node | value | class |
|---|---:|---|
| saving: −1.488 × 162.7 | **−242** | **measured** basis |
| cost: +11.8 cold weight-row lines (0.14147 × 4.18 rows × 20 lines) × 7.4 | **+106** | derived; cycles/line is §3.2's residual |
| **net** | **−136** | bracket **[−183, −89]** on the cost's 5–13 cycles/line band |

**−136 cycles/node = 1.034× = +4.7 Elo.** Real, buildable, and a tenth of what
study 28's #1 ranking implied. It also has one **unquantified credit** in its
favour, stated in §3.4.

---

## 3. THE MODEL — a per-node line budget for the whole engine

### 3.1 Construction

Every NNUE memory access is a 64-byte line from one of four places. The count
of lines is **geometry × a measured rate**; the price of a line is measured for
one class and a residual for the other.

Geometry at `ACC = 1280`: an accumulator is 2560 B = **40 lines** per
perspective; a HalfKA weight row is 1280 × 2 B = **40 lines**; a threat weight
row is 1280 × 1 B = **20 lines**. Stockfish at 1024: 32, 32, 16.

| class | lines/node | how |
|---|---:|---|
| HalfKA fold, parent read + child write | 78.0 | 0.9749 folds × 80 |
| threat fold, parent read + child write | 68.0 | 0.8498 folds × 80 |
| `activate` reads 4 accumulators | 68.9 | 0.4307 evals × 160 |
| Finny refresh: entry read, entry write, stack write | 19.3 | 0.16102 × 120 |
| threat refresh `dst.fill(0)` | 0.8 | 0.01955 × 40 |
| **accumulator streaming subtotal** | **235.0** | |
| HalfKA fold rows (57.7 MB table) | 88.7 | 2.218 rows/n × 40 |
| HalfKA refresh rows (57.7 MB table) | 36.5 | 0.9132 rows/n × 40 |
| threat fold rows (77.7 MB table) | 78.2 | 3.912 rows/n × 20 |
| threat refresh rows (77.7 MB table) | 1.6 | 0.082 rows/n × 20 |
| **weight-row subtotal** | **205.0** | |
| **total FT traffic** | **440.0 lines = 28.2 KB/node** | |

### 3.2 Pricing the lines

Accumulator streaming: **2.36 stall cycles/line, measured** (§2.4).
Non-NNUE floor (TT, history, movegen, board): study 28's PSQT arm stalls 712.7
cycles/node; scaled from its PGO frame to mine (×1.216) ≈ **867**.
Weight rows: **the residual**, (2934.35 − 867 − 235.0 × 2.36) / 205.0 =
**7.4 cycles per cold line**.

⚠ **The residual absorbs every modelling error**, including the head's weight
reads and the PSQ1 rows, which I have not counted separately. Treat 7.4 as a
band of **5–13** and the two weight-row rows below as ceilings, not
measurements. The *line counts* are solid; the *cycles per line* for the cold
class is not.

### 3.3 The result

| class | lines/node | cycles/line | load-stall cycles/node | share |
|---|---:|---:|---:|---:|
| non-NNUE (TT, history, movegen, board) | — | — | ~867 | 30% |
| **HalfKA weight rows** | 125.2 | 7.4 *(residual)* | **926** | **32%** |
| **threat weight rows** | 79.9 | 7.4 *(residual)* | **591** | **20%** |
| accumulator streaming | 235.0 | **2.36 *(measured)*** | **555** | 19% |
| total | | | 2939 (measured 2934.35) | |

**Read this against where thirteen studies have been looking.** The threat
channel has had four studies to itself; it is 20%. Accumulator streaming — the
whole fusion debate — is 19%. **The HalfKA weight rows are 32% and have never
been the subject of an arm.**

It also explains three standing facts that had no shared explanation:

* **Why prefetch is dead** (five refutations, the last against a
  wrong-address control at +0.01%). 205 lines/node from 135 MB of tables
  against a 32 MB L3 are compulsory misses at a memory-level parallelism the
  out-of-order window already saturates. There is nothing to prefetch away.
* **Why our miss RATE equals Stockfish's** (19.95% vs 19.68%). The dominant
  class is compulsory and structurally identical in both engines.
* **Why width's measured elasticity is 0.31, not 1.0.** Weight-row bytes scale
  linearly with width, but at 7.4 cycles per line they are ~85% overlapped
  against a ~50-cycle L3 latency. Width buys bytes; the machine only charges
  for the ones it cannot hide.

### 3.4 The same budget applied to Stockfish over-predicts it by 35% — and that is a finding

Stockfish's line counts, same method (fold rate 0.8654, refresh 0.1673, row
census from study 12: 7.9484 rows/node against our 7.125):

* accumulator streaming ≈ **102 lines/node** against our 235 — **2.30×**.
  Study 12 measured the same ratio directly and by a different route
  ("accumulator passes/node 2.3025 vs 1.1655 ⇒ **2.47× lane-copies**"); 2.30
  against 2.47 is 7% apart, which is as close as two independent derivations
  from different trees get, and it is the model's only external check.
* weight rows ≈ **183 lines/node** against our 205 — only **1.12×**, because
  their rows are 1.25× smaller in bytes but they fold slightly more of them
  and rebuild all threats on every refresh.

Pricing those at our constants gives ~2,460 stall cycles/node against
Stockfish's measured ~1,845 (its 1517.5 PGO figure scaled to my frame).
**The model over-predicts Stockfish by 33%** — and since its line counts are
solid, the over-prediction has to land on the price. Solving for it gives
Stockfish **~4.3 cycles per cold line against our 7.4**. ⚠ This is a residual
of a residual and it is the weakest number in the document; it is quoted
because its *sign* is corroborated independently, not because 4.3 is a
measurement.

That is consistent with the one hierarchy statistic that is not a wash:
**L3 fills 1.64× and DRAM fills 1.46×**, against an L1 miss ratio of only
1.34×. Our misses reach further down. The most likely mechanism is that our
accumulator stack — **10.56 KB/ply against their 4.16, ~160 KB of live working
set against ~70 KB in a 1 MB L2** — evicts weight rows that would otherwise
have been L2 hits.

**This is the strongest remaining argument for fusion and it has never been
made.** Fusion halves the accumulator footprint, and the credit is *not* the
−242 cycles of its own traffic (§2.4) but an unmeasured reduction in the depth
of everyone else's misses. It is also precisely what the §2.4 probe cannot
see, because adding a copy adds pressure symmetrically. **If the 7.4 → 4.3
cycles/line gap is footprint-driven, fusion is worth −136 plus up to
205 × 3.1 = −636 more (**+29 Elo in total**), and it becomes item #1 by a distance.** I did not
measure it; §4.4 says how.

---

## 4. THE THREE MEASUREMENTS THAT SHOULD HAPPEN NEXT, AND WHY EACH

### 4.1 A co-access census of the two FT tables — prices items 1 and 2

**The question:** within one fold, and within one refresh, which feature
indices fire together? Our HalfKA fold touches 2.275 rows and our threat fold
4.60; each row is a random 1280-or-2560-byte slice of a 57.7 MB or 77.7 MB
table. If co-accessed rows were adjacent in memory, a single line fill would
serve several of them.

**Why it is the right next thing:** it is the only lever that attacks
items 1 and 2 *without a retrain*. A permutation of the feature index space is
**bit-identical** — apply it in `sota_features.rs` (the index) and
`sota_loader.rs` (the row order) and the engine computes exactly the same
sums. Study 25 §8.5 named it and called it "a research task, not a perf arm",
which was right and is why nobody has done it; the census is the part that
makes it an arm.

**Cost:** one instrumented build (the same shape as §2.1's) emitting the
resolved index list per fold, plus a histogram. Half a day. **Prize:** up to
1,517 cycles/node sit behind it (§3.3); a permutation that converts 20% of
those fills into L2 hits is ~300 cycles/node ≈ **+11 Elo**.

**Falsifier, stated in advance:** if the census shows co-access is
approximately uniform over the index space — i.e. the rows that fire together
in one fold have no stable relationship — the permutation cannot exist and the
item is dead. Record that as a result and move to §4.3.

### 4.2 Localise `search::<NonPv>`'s +303.4 cycles/node — item 3

Study 28 measured it with `perf record` on the stall event itself, so the
number is sound. §2.2 killed its only named mechanism. What remains inside
that symbol: the TT probe (16 MB, random), the butterfly history plane
(256 KB) and the continuation-history slabs, movegen, and the HalfKA
`push`. **Stockfish's equivalent symbol stalls 149.2.**

**Cost:** one build with four `#[inline(never)]` boundaries and one
`perf record`. Two hours. This is study 21's own house method — "an
instrumented probe instead of an arm" — which killed the event-cancellation
lead for the price of one build.

**Caveat to carry:** the two engines' search/eval symbol boundaries are not the
same (study 28 §2), so part of the 3.03× is Stockfish splitting out
`update_all_stats` and friends. Grouping search+movepick gives 1.527×, not
3.03×. **The honest ceiling on item 3 is therefore below −303 and nobody
knows by how much** — which is itself a reason to measure rather than to plan
around it.

### 4.3 The i8 HalfKA row — the largest single item in this document

**Mechanism:** our HalfKA weight rows are `i16` — 2560 B, **40 lines**, from a
57.7 MB table. Our threat rows are already `i8` — 1280 B, 20 lines, from a
77.7 MB table, accumulated with `vpmovsxbw` + `vpaddw` at two instructions per
32 lanes. Doing the same to HalfKA halves the largest cold class:

> 125.2 lines/node → 62.6. At 7.4 cycles/line that is **−463 cycles/node =
> 1.101× = +17 Elo** — larger than fusion and width combined, and it is
> orthogonal to both.

It also halves the table to 28.9 MB, which brings it **inside** the 32 MB L3
for the first time; the residual 7.4 cycles/line would itself fall. And it
composes with width: at `ACC = 1024` with i8 rows the HalfKA table is 23.1 MB.

**What it costs:** accuracy. FT weights at the current QA = 255 frame exceed
`i8` range, so this is a quantisation-aware-training question, not a loader
change — the same class of change as the threat channel's existing 64 frame.
Stockfish has not done it, which is evidence it is hard, not evidence it is
impossible: Stockfish's table is 111 MB and it has the same L3.

**Cost to find out:** zero engine work. Quantise an existing D48X checkpoint's
FT to i8 offline and run the 800-FEN fidelity battery
(`tools/nnue/net_scale_check.sh` is already the pre-game gate for scale
changes). If the mean SF-cp error is in the same range as the threat channel's
measured **0.14 SF cp**, the item is live. **A day, and it is the highest
expected value on this list.**

**Estimated, not measured**, and labelled so: the −463 uses §3.2's residual,
and the accuracy side is unpriced.

### 4.4 The one number that would most change the plan

**Is the 7.4-versus-4.3 cycles-per-cold-line gap against Stockfish caused by
our 2.5× accumulator footprint?**

If yes, fusion is worth up to −772 cycles/node (+29 Elo) instead of −136, and
it becomes item #1 immediately. If no, the gap is intrinsic to our larger
tables and only §4.3 and width can touch it.

**How to get it, and the design of the experiment matters** — the §2.4 probe
cannot answer it, because *adding* a copy adds both traffic and pressure and
the two cannot be separated by sign. The clean instrument is a
**footprint-only intervention**: keep every arithmetic operation identical and
shrink only the live stack. The cheapest form is to build fusion behind a
cargo feature and measure `ls_dmnd_fills_from_sys.*` — **L3 and DRAM fills per
node, not L1 misses** — on both sides. If the fill *depth* improves by more
than the fill *count*, the footprint mechanism is real.

**Coordinate, do not duplicate:** a sibling study (29) is measuring the fold
kernels' load split between accumulator streaming and weight rows. §2.4 and
§3.1 answer that question from a different direction — 235 accumulator lines
against 205 weight lines, with the accumulator side priced at 2.36 cycles/line.
**If study 29's split disagrees with 235/205, believe study 29** — it is
measuring the kernels directly and I am counting geometry.

---

## 5. THE DISCOUNT, DERIVED RATHER THAN ASSERTED

Studies 8–19 proposed a great many priced items. Tabulating every one that
carried both an ex-ante number and a post-hoc measurement:

| set | predicted i/n | realised i/n | ratio |
|---|---:|---:|---:|
| studies 8–13, all priced items (n = 22) | 5,206.4 | 406.2 | **7.8%** |
| studies 14–19, all priced items (n = 13) | 425.5 | 31.8 | **7.5%** |
| **combined, all priced** | **5,631.9** | **438.0** | **7.8%** |
| combined, **landed subset only** | 3,152.5 | 689.3 | **21.9%** |

But the aggregate hides the only pattern that is actually useful for
forecasting, and it is a clean split:

| how the item was priced | predicted i/n | realised i/n | **ratio** |
|---|---:|---:|---:|
| **change-specific** — "this edit removes N instructions at M sites", from a static count or a stub-and-diff | 496.0 | 379.3 | **76.5%** |
| **block-excess** — "we are N i/n worse than Stockfish at this block" | 2,335.0 | 158.0 | **6.8%** |

*(Studies 8–13's landed items, split by pricing method. Study 14–19's landed
subset independently gives 47–50%, sitting between the two.)*

**A block-excess price is not a price.** It measures how much work happens
somewhere, not how much of it is removable — and studies 9, 11, 12 and 13
between them closed the head, movepick, make/unmake, `at_full` and the fold
kernels *at parity* after each had been carried for several studies as a
several-hundred-i/n "opportunity". One more datum in the same direction: study
10 priced Stockfish's small-net advantage at 325–740 i/n and study 11 measured
it at **197.7**, over-priced 1.6–3.7×.

**Study 28's four ceilings are all block-excess prices**, and the two I could
test came in below even the 6.8% rate:

| study 28 item | its price | this study |
|---|---:|---|
| A — width 1280 → 1024 | −271 | stands: a **measured intervention**, not a block excess |
| B — fuse the accumulators | −150 … −375 | **−136**, measured basis (§2.4) |
| C — the per-node snapshot | −264 | **0** (§2.2) |
| D — front-end + retiring + other | −196 | untested; at 6.8% it is −13 |

Study 28 itself flagged that its four ceilings summing to exactly Stockfish
parity (3000.8 against 3005.0) was "far too tidy to believe". It was right to,
and this is how it fails: not by all four shrinking a little, but by one of
them being zero.

**One reason to expect better than the historic rate, and it is real.** Every
one of those 20 landed items was aimed at instruction count, which study 28
showed is not the binding constraint. At our IPC a removed instruction is
worth **0.78 cycles**; a removed load-stall cycle is worth **1.00**, and §2.4
measured a class where the exchange is far better than that — the extra-copy
probe cost 133.4 instructions and 162.7 stall cycles, i.e. **1.22 stall cycles
per instruction**, 1.56× the average. Aiming at lines rather than instructions
should raise the realisation rate. It has not been demonstrated yet, and I am
not counting it in the forecast.

**Forecast.** Applying the derived rates — measured interventions at ~80%,
block-excess items at 6.8%, the one measured item at its measured value:

```
item 4 (fusion, measured basis)            -136 x 0.80  = -109
item 5 (width, measured intervention)      -271 x 0.80  = -217   [Elo-neutral, see 6.2]
item 6 (front-end/retiring, block-excess)  -196 x 0.068 =  -13
items 1-3 (unpriced ceilings, 1820 total)  x 0.068              = -124
                                                          -------
                                                            -463 cycles/node
4106.8 -> 3644  =  1.127x  =  +21 Elo   (of which width's +12 is Elo-neutral)
```

**Net expected value: +8 to +18 Elo**, landing at **1.15–1.25× Stockfish per
node**. Study 28's +22 to +32 assumed item C was real and item B was 2–3× its
measured size.

---

## 6. WHAT IS NOT WORTH DOING, WITH THE NUMBER

### 6.1 Further instruction-count micro-optimisation

We are at **1.2235× Stockfish's instruction count** and **better than
Reckless**, which executes 25.3% more instructions than we do and is
marginally faster. Thirteen studies bought −17.3% of cycles; the last one to
try (study 27) removed 1% of the engine's branches and the mispredict rate
**did not move** (+0.747% against a −0.280% null). Study 10 priced 1,667 i/n
and landed 62.8 — **3.8%**. At 0.78 cycles per instruction, a heroic further
−300 i/n is −234 cycles/node, and the record says −300 is not available.
**Closed.**

### 6.2 Narrowing to 1024 *for Elo*

The speed is real and measured: `(1280/1024)^0.306 = 1.0707×` = **+11.8 Elo**.
But the project's own training finding is that **width is ~Elo-neutral between
768 and 1536 at equal time** (+44 fn per doubling of width against −45 Elo of
equal-time speed). Those cancel by construction, so narrowing buys speed and
hands back eval quality. Study 28 listed it as item A *and* argued it was
Elo-neutral; both are right and the resolution is that **it belongs in a
per-node-speed roadmap and not in an Elo roadmap.**

The one argument for taking it anyway is the project's own D48X ruling —
promoted at **−9.1 [−34.4, +16.1] Elo** on the judgement that the eval is
stronger and *the search cannot yet exploit it*. If that ruling is true, the
eval half of the width trade is currently worth less than the speed half, and
narrowing is ≥ neutral. **That is a decision for the owner, not a measurement**,
and it should be made once rather than re-litigated per study.

Two further notes: study 28's elasticity was measured on a tree at 7168.3 i/n,
which is **more instruction-bound than today's 5362.4**, so 0.306 is arguably
a *lower* bound on the current engine. And the elasticity is a whole-engine
number — do not also apply the 1.25× row-byte ratio from §3.1 on top of it.

### 6.3 Software prefetch, in any form

Dead five times: studies 4, 21 (+0.01% against a wrong-address control, n=15),
22 (twice), 24. §3.3 explains it: our miss *rate* equals Stockfish's, the
dominant class is compulsory, and the out-of-order window already saturates
the available memory-level parallelism. **Do not re-open without a new
mechanism.**

### 6.4 Front-end and code size

Priced at **+56.5 cycles/node, 5.0% of the gap, ≈ +2.4 Elo** (study 28 §0.3).
Perf study 7 ranked this #3. It is a twentieth of the gap.

### 6.5 The threat channel as a whole

Perf study 7's "the threat channel costs 1.78× our speed" was corrected to
1.25–1.39×, and study 28 measured Reckless carrying a **larger** threat feature
space (66,864 against our 60,720) at the same cycles/node. §3.3 puts our threat
weight rows at 20% of load-stall against HalfKA's 32%. **Four studies have
worked this channel; it is not where the mass is.**

### 6.6 Re-litigating "is Rust the problem"

Four controls kill it (study 28 §0.5), the strongest being that our own
NNUE-off build posts the highest IPC in the panel (1.90) and a Rust engine
out-retires Stockfish (1.645 vs 1.459). PGO alone moves IPC ±18%, more than any
language effect. **Closed.**

---

## 7. DEPENDENCY ORDER

```
  [4.1 co-access census] ──► item 1 (HalfKA row locality)   ─┐
                        └──► item 2 (threat row locality)    │
                                                             ├─► a permutation arm
  [4.3 i8 FT fidelity check] ──► item 1' (i8 HalfKA rows) ───┘   (bit-identical)
            │
            └── composes with ──► item 5 (width 1024)  ──► needs a retrain either way

  [4.2 NonPv localisation] ──► item 3 (mechanism unknown)  ──► ??? 

  [4.4 fill-DEPTH experiment] ──► decides whether item 4 is -136 or -772
            │
            └── the experiment IS a feature-gated fusion build, so it is
                the first half of the item-4 implementation, not a detour
```

Three things follow:

* **Nothing on this roadmap is blocked on fusion**, and fusion is blocked on
  nothing. Study 26 said it was "blocked on the Finny cache key"; §2.3 shows
  the blocker is a *cost* (a full threat rebuild on every own-king move), not
  an impossibility — Stockfish ships exactly that design.
* **Items 1, 1′ and 5 all end at the trainer.** They should be bundled into one
  retrain rather than three, and the fidelity work (§4.3) has to precede it.
* **Item 3 has no downstream** until it has a mechanism. That is why it is a
  measurement in §0.3 and not an item in the build plan.

---

## 8. WHAT I AM LEAST SURE ABOUT

Ranked by how much it would move the plan.

1. **The 7.4 cycles per cold weight-row line is a residual, not a
   measurement.** It carries every modelling error in §3.1 — the head's weight
   reads, the PSQ1 rows, and any traffic I have not counted. Items 1, 2 and
   4.3 are all priced with it, so if it is really 5 they shrink by a third and
   if it is really 13 they grow by 75%. Study 29's fold-kernel load split is
   the nearest thing to a check and I have not seen it.
2. **The non-NNUE floor (867) is scaled across frames** — study 28's PSQT arm
   in the PGO frame, multiplied by my probe frame's cycle ratio. It is 30% of
   the budget and it is the crudest input in the model.
3. **Item 3's ceiling is not −303.** The two engines' search/eval symbol
   boundaries differ; grouping search+movepick gives 1.527× rather than 3.03×.
   I do not know how much of the +303.4 survives a like-for-like grouping and
   I did not try to find out.
4. **The 4.18 rows per threat rebuild is measured on `wl50`, which is
   endgame-heavy.** Study 21's threat *fold* rate on the deeper `bench16` frame
   is 7.75 rows against my 4.60 on `wl50` — a 1.68× frame difference. If the
   rebuild scales the same way, a middlegame rebuild is ~7 rows and fusion's
   cost term grows by 1.68×, taking the net from −136 to about −100. It does
   not change the sign.
5. **My probe frame is non-PGO, and the probe→PGO translation has gone wrong
   in both directions** — studies 12–15 over-stated by 1.12–2.0×, study 17 by
   2.17×, and study 18 *under*-stated by 2.94×. Every cycles and stall number
   I measured is a probe-frame number.
6. **The box was contended.** The gap itself is 1.37× quiet and 1.48× loaded,
   and my stall nulls (±2.1%, ±2.7%) are larger than they would be on an idle
   machine. The two arms that matter (§2.2, §2.4) rest on the L1-miss counter,
   whose null held at ±0.16% throughout, so they survive; nothing else I
   measured would.
7. **I did not run a game.** The standing bar says games arbitrate and a
   session with no completed game result produced nothing. This study produced
   one landed instruction-neutral layout change and two refutations, none of
   which is worth a match. **The first item on this roadmap that reaches an
   engine binary should be gated on games, not on `perf`.**

---

## 9. REPRODUCING THIS

```bash
bash tools/perf31/build31.sh  <label>            # fat LTO, CGU=1, native, sota+arch-d48x
bash tools/perf31/ident31.sh  <label>...         # i/n, br/n, node count, wl50 eval hash
bash tools/perf31/stalls31.sh <base> <arm> [n]   # randomised 3-way round-robin vs a
                                                 # byte-identical null; cycles, instructions,
                                                 # ex_no_retire.load_not_complete, L1d loads+misses
```

Raw panels: `docs/research/perf-artifacts-31/stalls.txt`,
`docs/research/perf-artifacts-31/ident.txt`.

The counter build of §2.1 was a throwaway (ten `Relaxed` counters in
`sota_features.rs`, `sota_accumulator.rs`, `sota_threats.rs`, printed from
`bench.rs`) and is **not** in the tree. Rebuild it from §2.1's counter list;
its four self-controls are stated there and any re-derivation should reproduce
them before its fifth number is believed.

**Gates — all four run, none asserted.**

| gate | pin | measured on arm p1 |
|---|---|---|
| `Bench:` depth-6 fingerprint, default features, `x86-64-v3` | 587683 | **587683** ✓ |
| NNUE gate, wl50 eval hash | `5ca978e0…8b38` | **`5ca978e012da1bb0`** ✓ |
| NNUE gate, d13 node total | 1922947 | **1922947** ✓ |
| `cargo test --features sota,arch-d48x` | — | 45 pass, 0 fail (4 suites) |

`cargo clippy` and `cargo fmt --check` are clean. One deviation to state
rather than bury: the d13 run used the `target-cpu=native` (AVX-512) build
rather than the gate's `x86-64-v3` (AVX2) one. Node counts and the eval hash
are backend-independent by design, and study 26 §4 reproduced both pins on the
AVX2 path explicitly; this arm changes no kernel, only where two fields live.
Arm p1 is additionally node-identical (5,001,562) and eval-hash-identical on
every panel in §2.2 and §2.4.
