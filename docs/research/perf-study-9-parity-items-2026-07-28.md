# ARCHON perf study #9 — the five carried parity items, implemented

Date: 2026-07-28. Ninth commissioned study in the chain (8:
`perf-study-8-nps-parity-2026-07-28.md`, whose §9 "what I would do next" list is
this study's mandate). Raw artifacts and instruments: `perf-artifacts-9/`.

**This was an implementation commission, not an evaluation.** Branch
`perf-study-9`, based on `perf-study-8-nps-parity`. Deliverable binary:
**`engines/bin/archon-s9-final`** (PGO, `sota,arch-d48x`, canonical net).
Baseline for every comparison: `engines/bin/archon-s9-base` (study 8's HEAD,
rebuilt here and verified to reproduce both study-8 pins exactly).

**Three changes landed, four items killed by a number, and the two overdue
measurements were taken on a genuinely idle box.** All three landed changes are
**bit-exact and node-identical**: `CHECK_NNUE_GATE=1 bash tools/check.sh` exits
0 with eval hash `5ca978e0…8b38`, d13 nodes `1,922,947` and `Bench: 587683`
unchanged, and the D24 perft ritual passes.

---

## 0. HEADLINE

### 0.1 The matched panel

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, three arms interleaved,
`taskset -c 4`, `nice -n 19`, 6 rounds. Raw: `counters-final.txt`.

| per node | base (study 8 HEAD) | **final** | SF18 | parity target |
|---|---:|---:|---:|---:|
| **instructions** | 6,546.0 | **6,429.9 (−1.774%)** | 4,329.6 | 4,677 |
| ratio vs SF18 | 1.5119× | **1.4851×** | — | 1.080× |
| residual vs parity | 1.400× | **1.3747×** | — | 1.000× |
| branches | 764.1 | 755.2 (−1.17%) | 394.7 | — |
| `instructions:u` spread | 0.0032% | 0.0034% | 0.0000% | — |

Cumulative across studies 8 + 9: **7,116.6 → 6,429.9 i/n**, i.e. **28.1% of the
original 2,439.6 i/n excess closed** (study 8 alone had closed 23.4%).

### 0.2 Item by item

| # | item | verdict | the number |
|---|---|---|---|
| 1 | `at_full`'s replay plumbing | **LANDED** | **−0.852%** — both perspectives resolved in ONE pass over the event buffer, sharing the perspective-independent per-event decode (§2.1). A second, larger attempt at the same item (deleting the per-push position snapshot) was **KILLED at +0.70%** (§3.1). |
| 2 | Unify the two threat systems | **KILLED** | The premise is false and the arithmetic is against it. SF18's threat log serves **only** the NNUE; its move ordering recomputes `threatByLesser` from scratch per quiet movelist. And our two systems together are **270 i/n against SF18's 289 i/n for the NNUE half alone** (§3.2). |
| 3 | make/unmake at 2.26× SF18 | **LANDED (partial)** | **−0.157%** — the child `StateInfo` built in the vector's tail instead of a 168-byte stack temp (§2.2). Two further candidates killed: the repetition-walk stride at **+0.042%** (§3.3), and `set_check_info`, **priced at 106 i/n (1.50%)** with laziness the only structural lever and a per-move read cost that would eat it (§3.4). |
| 4 | The accumulator `memmove` block | **LANDED** | **−0.392%** — a two-destination Finny kernel writes the cache entry and the stack slot from the same registers (§2.3). |
| 5 | `l2_tail` + `l3` | **KILLED** | The block is **67.7 i/n (0.96%)**, against SF18's ~61 i/n equivalent — **1.03×, at parity**. The one implementable idea (constant-bound halving tree) measured **exactly 0.000%**: LLVM already unrolled the runtime-strided loop (§3.5). |
| 6 | Idle-box counter panel + SMP scaling | **MEASURED** | Speed is **0.7014 [0.6833, 0.7078]** of SF18, cycles/node **1.4061×** — a real number, not study 8's band (§4.1). **SMP NPS scaling is at parity with SF18**: 10.45× vs 10.47× at 16 threads (§4.2). |

### 0.3 The two findings that were not on the list

**(a) `bench`'s `threads` argument is parsed and never used.** `src/bench.rs`
reads it into `params.threads`, and nothing consumes it. The first SMP curve
this study ran was therefore a flat line at 1.00× for 1/2/4/8/16 threads with
**byte-identical node counts at every point** — a perfectly convincing null that
was measuring nothing. Driven through UCI instead (`setoption name Threads`,
which does reach `ThreadPool::resize`), the pool scales fine. Anyone measuring
ARCHON's SMP through `bench` is measuring one thread.

**(b) The obvious probe for a copy is removed by dead-store elimination.**
Pricing the Finny 2560-byte copy by writing it twice measured **+0.006%** —
because the compiler deleted the duplicate. With `black_box` between the two it
measured **+1.45%**. The delivered fix was worth **0.39%**. So a `black_box`'d
doubling probe over-states — it blocks optimisations the real code keeps — by
roughly 3-4× on this box, twice (§3.6). It is still the right instrument for
*ranking*; it is not a forecast.

---

## 1. METHOD

Study 8's rules were followed and each earned its keep again:

* **Choose by counting, never by `perf annotate` share.** Annotate was used only
  to *locate* phases. It pointed at `vmovdqu64 0xc0(%rsp),%zmm1` at 58.30% of
  `make_move_with` — almost certainly skid — but the *static* round trip it sat
  on (build a 168-byte temp, reload it, store it) was real, and removing it paid
  (§2.2). The distinction that matters: the annotate percentage was not
  believed; the instruction sequence around it was read.
* **Price by stub-and-diff before designing.** Six pricing probes were run
  (`perf-artifacts-9/probes.txt`). Two of them changed the verdict on an item.
* **Bit-exactness as a gate, not a goal.** All three landed changes turned out
  to be bit-exact anyway, so `evaldiff.py` was never needed — no change in this
  study perturbs the eval, and the wl50 eval hash is byte-identical across all
  three commits. The fidelity budget stayed unspent.
* **Instructions carry every claim.** The box went from idle (load 0.5) to
  contended (load 8-10, another agent's `fastchess` match plus a second
  `cargo test`) about 25 minutes in and stayed there. The counter panel and the
  SMP curve were taken in that first window *deliberately*, which is why §4 has
  real numbers instead of a band.

### 1.1 The idempotent-double-call trick

Two of this study's prices come from calling a **pure or idempotent** function
twice and reading the delta. Unlike a stub it changes no behaviour, so the tree
is bit-identical and the price is exact rather than an upper bound:

* `set_check_info` is idempotent (it overwrites what it wrote) ⇒ calling it
  twice prices exactly one execution: **106.4 i/n**.
* `propagate_l2`/`propagate_l3` are pure ⇒ one extra `black_box`'d call each
  prices the tail: **67.7 i/n**.

For a function that *writes* rather than computes, the same trick needs
`black_box` or dead-store elimination erases it (§0.3b).

---

## 2. WHAT LANDED

Arm harness: `scripts/arm9.sh <name>` — PGO build, both identity pins, then
`instructions:u` on a fixed 5,001,562-node tree. Probe harness:
`scripts/probe9.sh <label>` — native non-PGO, same tree, for the pricing pairs.
Per-change deltas below are non-PGO probe pairs (`probes.txt`); the headline
−1.774% is the PGO matched panel.

### 2.1 Item 1 — resolve both perspectives in one pass (−0.852%)

`ThreatContext::compute` regenerates the perspective-agnostic event list once,
then walked it **twice** — once per perspective — because the two resolve to
different feature indices.

But everything before the perspective enters is shared. `ThreatResolver::index`
begins by decoding the attacker and victim into SF piece codes, and the
perspective is only the `^ swap` colour flip applied *after* that decode. So the
two-pass shape paid the event-field loads and both `sf_code` decodes twice.

`resolve_both` fuses the walk. `ThreatResolver` gains `index_decoded` — exactly
`index` with the codes already in hand — and `index` delegates to it so the two
cannot drift.

Bit-identical: same arithmetic per perspective, and both the index lists and the
T6c PSQT accumulate still see events in buffer order, which is what fixes the
i32 sum. The one-sided case (a king-crossed ply computes one perspective eagerly
at push) deliberately keeps calling `resolve_into` — folding a `need` test into
the fused loop would tax the common both-sided path with a branch per event.

**7,066.8 → 7,006.6 i/n = −0.852%.** The largest single change in the study.

### 2.2 Item 3 — the child `StateInfo` in the vector's tail (−0.157%)

```rust
states.push(StateInfo { copied: prev_copied, ..StateInfo::EMPTY });
```

reads as "push a mostly-constant struct". It is not. A struct expression whose
fields are not **all** compile-time constants is materialised as a 168-byte
stack temporary and then copied into the vector, so every made move paid build +
reload + store. Perf study #2 option E removed exactly this round trip for the
subsequent *field writes* — and left it in the push itself.

Pushing `StateInfo::EMPTY` (every field constant) lets the value be stored
straight into the slot; the one live field is then written through a `&mut` from
`split_at_mut`. Behaviour-identical by construction: every non-`copied` field is
dead until it is written later in the same function.

**7,105.7 → 7,094.6 i/n = −0.157%.**

### 2.3 Item 4 — a two-destination Finny kernel (−0.392%)

The cached HalfKA refresh needs the same 2×1280 i16 in two places: the Finny
cache entry (its memo for the next refresh with this king key) and the search
stack's accumulator slot (what the head reads). It got there by folding in place
and then assigning the stack slot — a 2560-byte `memmove`, and the dominant
member of the libc `mem*` block study 8 §3.3 named at ~2%.

`i16_deltas_dual` writes both homes from the registers the fold already holds:
one extra store per 32-lane block instead of a whole read+write pass. Added to
all three backends with the same loads, the same 8-register tile and the same
per-lane wrapping adds in the same order.

A new test pins **both** homes against the scalar reference on every backend the
CPU offers. It earns its place: a backend that wired the second store up wrongly
would desync the stack from the cache **silently**, and the two are only ever
compared at the next refresh with the same king key.

**7,094.6 → 7,066.8 i/n = −0.392%.**

`scripts/libcalls9.py` is the instrument that named the call sites. Study 8's
`libcalls8.py` keyed on the call target's symbol name and found nothing here,
because the calls appear as `call *0x…(%rip)` with the name only in the PLT-slot
comment; the study-9 version keys on that comment and recovers the size
immediate. The 2560-byte sites are: `ThreatContext::refresh` ×2 (root, cold),
`apply_at`/`push_sota_dirty` ×2 each (the king-crossed eager refresh's by-value
return), `finny_refresh_into` ×2 (**fixed here**), and `at_full`/`current_full`
×1 each — the last two being the null-move parent copies study 8 §7.10 measured
as memoizations that cost **+1.04%** to remove. Those were left alone.

---

## 3. KILLED BY A NUMBER

### 3.1 Item 1's larger attempt — deleting the per-push position snapshot (+0.70%)

The mandate's framing for item 1 named `build_deltas` "regenerating events from
a snapshot" as the target, and the snapshot is genuinely paid on **every made
move** to serve the ~37% of plies that reach a fold. The information is also
recoverable without it: the position at the ply being evaluated **is** the live
board (the same contract `HalfkaContext::current_full` already has), and every
ply below it differs from its child by exactly the child's `DirtyPiece`.

So it was implemented: `apply_at` stores only the dirty record, and `compute`
reconstructs the span's positions backwards from the live board with a new
`ThreatPosition::unapply`. Bit-exact, node-identical, all 673 tests green
including the deep lazy-fold and random-replay gates.

**It measured +0.70% instructions on the PGO panel** (5 interleaved rounds,
per-binary spread < 0.001%) and was reverted. The patch is kept at
`perf-artifacts-9/killed-item1-nosnap.patch`.

The mechanism is visible in the flat profiles either side: `push_sota_dirty`
disappears (−0.83 percentage points into `do_move`) and `at_full` grows **+1.43
pp**. The reason is the fold *span*. Laziness means a fold reaches back over
every ply that never evaluated, so the number of positions reconstructed per
node is the number of **folded plies** per node — which is close to the number
of pushes, not to the 0.37 evals/node the saving was budgeted against. The
snapshot was already nearly as cheap as it can be; the reconstruction adds a
`from_board` plus an `unapply` chain on top.

**Recorded so it is not re-litigated: a lazily-folded stack does not save work
by deferring per-ply state that the fold span will demand anyway.**

### 3.2 Item 2 — unify the two threat systems

The item rested on: *"SF18 runs ONE (`update_piece_threats`, 5.65%) that serves
both its ordering and its net. Ours together are 397 i/n against SF18's 304 for
the same two jobs."* **Both halves are wrong.**

**The premise.** A full read of the SF18 sources says its threat log serves
**only** the NNUE. `DirtyThreats` / `threatenedSqs` / `threateningSqs` have
exactly four consumers outside `position.*`, all under `nnue/`:
`full_threats.h`'s `DiffType`, `AccumulatorStack::push`, the `double_inc_update`
fusion test, and `append_changed_indices`. Move ordering computes its threats
**completely independently and from scratch, per quiet movelist**, via four
`pos.attacks_by<Pt>(~us)` calls in `movepick.cpp` — the same whole-army setwise
shape as our `compute_threats`. A case-insensitive grep for `threat` across
`search.cpp`, `search.h`, `history.h`, `movepick.h` and `evaluate.cpp` returns
only the `do_move` plumbing: no threat-keyed history, no threat term in pruning
or extensions. **The sharing this item wanted to copy is something SF does not
do**; if we ever achieve it, it is an advantage over SF18, not catch-up.

**The arithmetic.** Re-measured on this study's own flat profiles rather than
inherited:

| job | ARCHON | SF18 |
|---|---:|---:|
| NNUE threat event generation | `record_square_touch` 3.00% + `record_targets` 0.58% = **234 i/n** | `update_piece_threats<false,true>` 3.27% + `<true,true>` 2.04% + `swap_piece` 1.37% = **289 i/n** |
| move-ordering threat map | `ensure_threats` 0.55% = **36 i/n** (lazy, position-keyed, ~52% read rate) | inlined in `next_move`; recomputed per quiet movelist, not cached |
| **total for both jobs** | **270 i/n** | **289 i/n for the NNUE half alone** |

We are already **0.81× SF18** on the NNUE half, and our ordering map is a lazy
position-keyed cache where SF's is an uncached per-movelist rebuild. Study 8's
397 i/n priced `compute_threats` at its **pre-laziness** 2.1% rather than the
1.01% its own §4.3 had just delivered. Unifying would be work in the direction
of a shape that costs more.

### 3.3 Item 3's repetition-walk stride (+0.042%)

`StateInfo` is 168 bytes — not a power of two — so `states[child_idx - i]`
re-derives the byte offset with an `imul $0xa8` on every iteration of a loop
that can run up to `rule50/2` times. Rewritten as a reversed `step_by(2)` slice
iterator (same visit order, constant −336 byte stride, strength-reducible to a
pointer decrement), it measured **+0.042%** and was reverted: `end =
min(rule50, plies_from_null)` is small in a search — bounded by the distance to
the last null move — so the loop almost never runs enough iterations to
amortise the iterator's setup. The negative is recorded in the code.

### 3.4 Item 3's `set_check_info` — priced at 106 i/n, not attempted

`set_check_info` is 206 static instructions, runs once per made move, and prices
by idempotent double call at **106.4 i/n = 1.50% of the engine** — the single
largest block in make/unmake. SF does the identical work (`update_slider_blockers`
for both colours, `checkSquares` for one king only — which we already match), so
there is no shape to copy.

The only structural lever is **laziness**, on the §4.3 model: the blockers,
pinners and check-squares are read only by nodes that generate or validate
moves, and nodes that return before trying a move (TT cutoffs, qsearch
stand-pats, razoring/futility returns) never touch them. The reason it was not
attempted: unlike the Wave-G threat map, whose two consumers read it **once or
twice per node**, this state is read **per move** — `gives_check` consults
`check_squares` and `blockers_for_king` for every move the parent tries. A
position-keyed cache check on each of those reads plausibly costs more than the
1.50% it saves, and making the fill coarse-grained instead means threading an
`ensure` through movegen, `legal()` and `gives_check` — a real refactor with a
real regression surface.

**This is the one item on the board with a measured price and no measured
verdict.** It is stated that way deliberately: 106 i/n is the ceiling, the read
rate is the unknown, and the next agent should measure the read rate (count
nodes that reach movegen vs nodes that make a move) before writing any code.

### 3.5 Item 5 — `l2_tail` + `l3` is at parity (0.96%, and the fix measured 0.000%)

Priced by an extra `black_box`'d call to each: **67.7 i/n = 0.96% of the
engine** for the whole 16→32→1 tail. Study 8's estimate was 2.16%; the real
figure is well under half that, and against SF18's `fc_1`+`fc_2` at ~1.4% of its
engine (≈61 i/n on the §1 tail assumption) we are at **1.03× — parity**.

The one implementable idea was `propagate_l3`'s reduction, written as
`while stride > 0 { for k in 0..stride { sums[k] += sums[k+stride] } }` — a
runtime trip count over a `[f32; 16]`, which looks like it must spill. Rewritten
with constant bounds (identical pairing, bit-identical result) it measured
**35,344,945,289 → 35,344,934,066 instructions = −0.00003%**. LLVM had already
unrolled the strided loop, because the stride sequence is statically known. The
change was reverted as pure churn.

Study 3's chain-split half was already rejected. **Item 5 is closed: the block
is smaller than believed, at parity with the reference, and its one remaining
idea is a measured null.**

### 3.6 The two probes that lied, and how

| probe | said | delivered | why |
|---|---:|---:|---|
| Finny copy, doubled assignment | +0.006% | — | **Dead-store elimination** removed the duplicate. A doubling probe on a *write* is invalid without `black_box`. |
| Finny copy, `black_box`'d | +1.45% | **0.39%** | `black_box` blocks optimisations the real code keeps, so it prices the copy *plus* the optimisation barrier. |
| `resolve_into`, `black_box`'d | +7.2% | **0.85%** | Same, and the probe also stood up two 2 KB scratch arrays the real code does not. |

Both `black_box` prices over-stated by 3-8×. They still **ranked** correctly —
`resolve_into` was the biggest block in `at_full`'s plumbing and the fix was the
study's biggest win — which is the job study 8 assigns to a pricing probe
("price first, design second"). Read them as a ranking, never as a forecast.

---

## 4. THE TWO OVERDUE MEASUREMENTS (item 6)

Both were taken in the session's first 25 minutes, while the box was genuinely
idle (load 0.2-0.5, 5 GB free). It did not stay that way — from ~19:45 it
carried another agent's `fastchess` match plus a second `cargo test`, load 8-10.
**Taking them early was the difference between a number and a band.**

### 4.1 The idle-box counter panel

8 interleaved rounds, `taskset -c 4`, `nice -n 19`. Raw: `counters-idle-base.txt`.

| per node | ARCHON D48X PGO | SF18 | ratio |
|---|---:|---:|---:|
| instructions | 6,546.0 | 4,329.6 | 1.5119× |
| **cycles** | **3,896.0** | **2,770.7** | **1.4061×** |
| branches | 764.1 | 394.7 | 1.9361× |
| branch-misses | 22.4 | 26.1 | 0.8587× (we win) |
| NPS (median) | 1,376,486 | 1,959,946 | — |
| **paired NPS ratio** | — | — | **0.7014, range [0.6833, 0.7078]** |

Study 8 could only offer "0.686, range [0.600, 0.736]" — a 23% spread. **The
idle-box range is 3.6% wide.** Instructions reproduced study 8's 6,546.0 and
SF18's 4,329.6 exactly, which is the fifth independent session to land SF18 on
that four-digit figure.

Read plainly: **we run at 0.70× SF18's speed and burn 1.41× its cycles per
node**, against 1.51× its instructions — so our IPC is *better* than SF18's
(the branch-miss ratio of 0.86 is the same story). The deficit is work done, not
work done badly.

### 4.2 SMP scaling — at parity with SF18

`setoption name Threads`, `go depth 22`, two positions, 3 rounds, idle box. Raw:
`smp-uci.txt`. **NPS speedup relative to each engine's own 1-thread figure:**

| threads | ARCHON (startpos / kiwipete) | SF18 (startpos / kiwipete) |
|---:|---:|---:|
| 2 | 2.09 / 2.01 | 2.09 / 2.07 |
| 4 | 4.04 / 4.06 | 3.95 / 3.90 |
| 8 | 7.70 / 7.87 | 8.16 / 7.83 |
| **16** | **10.45 / 10.15** | **10.47 / 10.32** |

The box is an 8-core 7700X with SMT, so ~10.4× at 16 threads is the SMT ceiling
and **both engines sit on it**. Time-to-depth at a fixed depth is noisy and
sub-linear for both (helper threads widen the tree — node inflation reaches
20-22× at 16 threads for both engines); that is ordinary Lazy SMP behaviour and
not a discriminator at these depths.

**Verdict: SMP is not a deficit. It has been flagged UNMEASURED since study 7
and the answer is parity.**

The caveat that makes this measurement non-trivial is §0.3a: **`bench` ignores
its `threads` argument.** The first curve, run through `bench`, showed 1.00×
scaling at every thread count with byte-identical node counts — a clean-looking
null that was measuring a single thread five times.

---

## 5. UNCERTAINTIES AND THREATS TO VALIDITY

1. **Speed numbers for the FINAL binary are a band, not a claim.** The box was
   contended for the entire second half of the session. §4.1's numbers are the
   *baseline* on an idle box; the final arm's −1.774% rests on
   `instructions:u`, which reproduced at 0.0034% spread. At the house ~1 Elo per
   1% NPS the landed set is worth roughly +1.8% NPS ⇒ ~2 Elo, offered as an
   instruction fact and not an Elo claim.
2. **Nothing here was Elo-tested**, and nothing here changes the eval, so the
   fidelity gate (`evaldiff.py`) went unused. All three commits are bit-exact.
3. §3.4's `set_check_info` verdict is a **price without a verdict** — the read
   rate that decides it was not measured. Stated as such.
4. The per-change deltas in §2 are non-PGO probe pairs; the headline is the PGO
   matched panel. They agree in sign and roughly in magnitude (−1.394% non-PGO,
   −1.774% PGO) but are not the same measurement.
5. §3.2's SF18 consumer analysis is a source read, not a measurement. The
   *arithmetic* half of that kill (270 vs 289 i/n) is measured on both engines'
   flat profiles from this session and stands independently.

---

## 6. WHAT IS LEFT, IN ORDER

The residual is **1,752.9 i/n over the 4,677 parity target**. After two studies
of implementation the honest shape has sharpened: there is no single lever
anywhere on the board worth more than ~1.5%, and two of the four items this
study inherited turned out to be at or better than parity already.

1. **`set_check_info` laziness — 106 i/n priced, read rate unknown** (§3.4).
   The only item left with a measured price and no verdict. Measure the read
   rate first.
2. **The eval layer's remaining excess.** After this study the eval layer is
   still where the bulk sits, and the two biggest named mechanisms
   (quantisation frame, skip lane) were already taken in study 8.
3. **Not the search shell** (1.16×), **not make/unmake's structure** (SF does
   the same work), **not the threat systems** (§3.2), **not the head tail**
   (§3.5), **not SMP** (§4.2). Five directions are now closed with numbers,
   which is worth as much as the 1.774%.

---

## 7. ARTIFACTS (`perf-artifacts-9/`)

* `counters-idle-base.txt` — the 8 idle-box interleaved rounds (§4.1).
* `counters-final.txt` — the final 3-arm matched panel (§0.1).
* `smp-uci.txt` — the SMP curve (§4.2). `smp-scaling.txt` is the **invalid**
  bench-driven first attempt, kept as the evidence for §0.3a.
* `probes.txt` — every pricing probe and per-change delta, in order.
* `arms.txt` — PGO arms with both identity pins.
* `flat-{s9base,s9nosnap,sf18}-instructions.txt` — flat instruction profiles.
* `killed-item1-nosnap.patch` — §3.1's reverted implementation, in full.
* `objdump-s9base.txt` — disassembly used for the static counts.
* `scripts/`
  * `arm9.sh` / `probe9.sh` — the PGO arm and non-PGO pricing harnesses.
  * `smpuci9.py` / `smpparse_uci.py` — the UCI-driven SMP curve (**use these;
    `smp9.sh`/`smpparse9.py` drive `bench` and are kept only as the record of
    the trap**).
  * `libcalls9.py` — libc `mem*` call sites by enclosing symbol and size,
    keyed on the PLT-slot comment rather than the symbol name.
  * everything else is inherited from `perf-artifacts-8/scripts/`.

**Reproduction.**

```bash
# identity pins (both unchanged from study 8)
./engines/bin/archon-s9-final nnue-eval engines/canonical/net.cenn \
  < tools/nnue/calib/wl50.fens | sha256sum        # -> 5ca978e0…8b38
taskset -c 4 ./engines/bin/archon-s9-final bench 16 1 13 \
  tools/nnue/calib/wl50.fens depth engines/canonical/net.cenn   # -> 1,922,947

# the matched panel
bash docs/research/perf-artifacts-9/scripts/pstat8.sh 6 300000 \
  tools/nnue/calib/wl50.fens /tmp/counters.txt \
  "final:./engines/bin/archon-s9-final @BENCH@ engines/canonical/net.cenn" \
  "sf18:./engines/bin/stockfish18 @BENCH@"
python3 docs/research/perf-artifacts-9/scripts/parse8.py /tmp/counters.txt sf18

# the SMP curve — through UCI, never through bench
python3 docs/research/perf-artifacts-9/scripts/smpuci9.py 3 22 /tmp/smp.txt \
  "ours=./engines/bin/archon-s9-final,engines/canonical/net.cenn" \
  "sf18=./engines/bin/stockfish18"
python3 docs/research/perf-artifacts-9/scripts/smpparse_uci.py /tmp/smp.txt
```
