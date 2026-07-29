# ARCHON perf study #19 — `at_full` is finally split, and the fattest stage turns out to be the one already at parity

Date: 2026-07-29. Nineteenth commissioned study (18:
`perf-study-18-lazy-king-cross-2026-07-29.md`, 17:
`perf-study-17-exact-counters-2026-07-29.md`). Raw artifacts and instruments:
`perf-artifacts-19/`. Baseline: `main` at **`60359e3`** — studies 12–18 all
merged and gated. Comparison target: **the `noinline` Stockfish `eca43a97`
rebuild** study 18 left at `/home/nick/archon-sf17-eca43a97`.

---

## 0. HEADLINE

### 0.1 What landed

| # | change | probe Δ i/n | probe Δ br/n | where |
|---|---|---:|---:|---|
| g2 | `DeltaBuf::push_kept` — stop re-testing `threat_pair_kept` at the 8 of 12 recording sites that have **already** established the pair is kept | **−28.46** | −1.654 | `sota_threats.rs` |
| g3 | `record_targets::<const KEPT: u8>` — the victim mask becomes a const parameter, so the `PieceType::ALL` scan disappears | **−9.34** | −3.531 | `sota_threats.rs` |
| g4 | the knight incoming scan's `threat_pair_kept(Knight, …)` guard is **dead** inside its own enclosing `if`; plus two more `push_kept` sites in `record_incoming_mutate` | **−7.61** | −1.174 | `sota_threats.rs` |
| | **total, probe frame** | **−45.41 (−0.738%)** | **−6.359 (−0.98%)** | |
| g1 | *(rejected, +4.63)* pre-masked `blockers` = `qatt & occ & !kings`, SF's `occupiedNoK` | — | | §6.1 |
| g1b | *(rejected, +1.82)* the king mask alone, without the hoist | — | | §6.1 |

Probe frame: thin LTO, 4 CGU, `-C target-cpu=native`, non-PGO, fixed
**5,001,562**-node tree. **Node count and eval hash identical on all ten lines**
(`5ca978e012da1bb0`), rejected arms included.

### 0.2 The commissioned target, closed

Study 17 item 1b's second half — owed since study 17, top-ranked by study 18 —
was: **`#[inline(never)]` on `compute`, `resolve_both`, `build_deltas` and the
fold, re-run at the 30k budget, price the four stages.** Done. Callgrind frame,
AVX2 `x86-64-v3`, 30,000 nodes × 50 positions, **1,500,780 nodes**:

| stage of `at_full` | self i/n | calls/node | i/call |
|---|---:|---:|---:|
| **`resolve_both` — per-event index resolution, both perspectives** | **285.65** | 0.5776 | 494.5 |
| `compute` — the ancestor walk + the child loop | 137.86 | 0.4315 | 319.5 |
| `build_deltas` — event generation (orchestration only; `record_square_touch` was already its own symbol) | 81.32 | 0.6182 | 131.5 |
| `fold_one` — phase-2 fold (the SIMD kernel was already its own symbol) | 50.51 | 1.1959 | 42.2 |
| `resolve_into` — the one-sided path | 4.24 | 0.0406 | 104.5 |
| **sum** | **559.58** | | |
| *unsplit `at_full` self, same frame* | *477.45* | *0.4315* | *1106.6* |
| *whole-binary total, base → split* | *15381.5 → 15467.0* | | ***+85.5 (+0.56%) — the outlining tax*** |

The +85.5 whole-binary move is the bound on the distortion, exactly as study 18
§4.1 bounded its Stockfish `noinline` rebuild at +7.5. It is the same size as the
gap between the stage sum (559.58) and the unsplit self (477.45), which is the
cross-check.

### 0.3 What this study is actually about

**(a) The fattest stage is the one we should NOT spend against, and only a
matched-frame Stockfish profile could show that.** `resolve_both` is 285.65 i/n,
51% of the split — and the obvious next move is to attack it. That would have
been wrong. Put both engines in the *same* 30k frame (study 18's SF numbers were
3k) and resolution is **1.076× SF per perspective-resolve**. The 1.289× total is
almost entirely a **call-count** difference that is a deliberate design trade,
and the trade is a wash: §4.

**(b) Generation is the block, and it is 1.52× per call.** ARCHON 367.53 vs SF
283.48 (1.30×); `record_square_touch` 186.9 i/call vs `update_piece_threats<true>`
122.6. Closing the per-call gap alone would put our generation column *below*
Stockfish's, because we already call the generator at 0.49× their rate. §4.3.

**(c) A whole family of "hoist it out of the slider loop" arms is dead, with the
number that kills it.** `ripple`'s body runs **0.481 times per node against
`record_square_touch`'s 1.0865** — **0.443 ripples per touch**. Anything moved
from per-slider to per-touch is paid 2.26× more often. That is why g1, which
takes a real Stockfish structural guard and removes real work, still measured
+4.63. §6.1.

**(d) The second metric axis fails its own null, pairing only half fixes it —
and it still earns its keep, by refuting one of this study's own hypotheses.**
Two builds of **bit-identical source** measured 4425.7 and 4810.5 cycles/node —
**+8.7%** — while `instructions:u` was identical to the digit. Pairing removes
the bias but leaves a median that wanders ±1.6% at n = 20, so no arm this study
landed is resolvable on it (§2, §5.1). What the axis *can* do is rank: a
`cache-misses:u` self-profile puts **65.75% of all cache misses in the three
accumulator-fold kernels**, and kills a clean-looking hypothesis this study had
already written into its own remainder. §5.2.

**(e) The comparative method has a structural blind spot, and it has been
steering the whole study program.** Ranking by diffing blocks against Stockfish
cannot see a symbol with no Stockfish counterpart. Self-profiles show
`finny_refresh_into` at **9.15% of cycles and 0.49% of instructions** — in no
study's ranked remainder — while `record_square_touch`, which has a clean
counterpart and was ranked second, is **1.98%**. §8.0.

---

## 1. METHOD

Studies 8–18's rules were followed. Study 18 §9.1's three handover rules were
live throughout; rule 3 ("an estimate carries its frame") is why §4 exists at
all.

### 1.1 NULL FIRST — four legs, all before any arm was written

**Null A — rebuild determinism** (same source, forced recompile):

```
nullpair-A   30,765,511,676   6151.1807 i/n
nullpair-B   30,765,510,917   6151.1806 i/n     759 apart, ±0.00015 i/n
```

**Null B — SOURCE PERTURBATION, structurally inert.** A reborrow
(`let r = &*r;`) at **exactly the two sites arm g1 was about to change** —
`record_square_touch` and `ripple`:

```
n0-NULLB-reborrow 30,765,510,879  6151.1805 i/n   797 from null A, ±0.00016 i/n
```

**Largest null on the instruction axis: 0.00016 i/n.** The smallest arm this
study keeps (g4, −7.61) is **48,000×** it; the smallest it rejects (g1b, +1.82)
is 11,000×. Nothing here is noise in either direction.

**Null C — the SEQUENTIAL cycles instrument. IT FAILS.** §2.

**Null D — the PAIRED cycles instrument, at n = 8 AND at the arm's own n = 20.**
§2.2, §5.1. The n = 20 leg is the one that matters, and it was run *after* the
arm because the arm's median looked too good — which is the wrong order and is
recorded as such in §10.

### 1.2 Instruments

* **`probe19.sh`** — screening, inherited from `probe18.sh`. Same frame. Two
  changes, both forced by the null panel: i/n printed to **four** decimals (one
  decimal cannot state a ±0.00015 null), and the built binary is **stashed** at
  `bin/<label>` so `pair19.sh` can come back for the contended axes.
* **`pair19.sh` — NEW.** The second metric axis, interleaved. §2.
* **`nullpair19.sh` — NEW.** The null for `pair19.sh`: builds identical source
  twice and pairs it against itself.
* **`split19.py` — NEW.** The measurement-only stage split of §0.2, with a
  `--revert`. It is **not** a shipping change and is not in the landed tree.
* `cg19.sh` / `cgsum19.py` / `cgbuild19.sh` — study 17's exact per-function
  counters, path-adjusted.

Every build inside its own `systemd-run --user --scope -p MemoryMax=3G
-p MemorySwapMax=0` cgroup and through `tools/oomlast.sh`; `probe19.sh` refuses
below 2600 MB available. A GPU training run held ~5.8 GB and 145% CPU for the
whole session; **it was never touched and nothing was OOM-killed.**

### 1.3 One instrument bug found and fixed before it could mislead

`probe19.sh`'s first version reported cycles as a min-of-N and printed a
per-arm IPC. That is the shape the commission asked for and it is **wrong on
this box** — §2 is the measurement that says so. It was replaced before any arm
ran, which is the only reason this study does not contain a reversed cycles
verdict.

---

## 2. THE SECOND METRIC AXIS — WHAT IT CAN AND CANNOT DO HERE

The commission requires `cycles:u` and `cache-misses:u` alongside
`instructions:u`, because instruction parity alone still leaves us ~1.16×
slower. That objective is right. The **instrument** needed work.

### 2.1 The sequential form fails its null

Two builds of bit-identical source, measured the way studies 12–18 measure
instructions:

| | i/n | cyc/n | cache-miss/n | IPC |
|---|---:|---:|---:|---:|
| null-1 (min of 3) | 6151.2 | **4425.7** | 82.77 | 1.390 |
| null-2-rebuild (min of 7) | 6151.2 | **4810.5** | 86.31 | 1.279 |
| | *identical* | **+384.8 (+8.7%)** | +4.3% | −8.0% |

**More repeats made it worse, not better** — null-2 had seven and a higher
minimum — because the interference is a GPU trainer whose load drifts on a
timescale longer than one probe run. This is rule 1's gate firing. Reading a
verdict off this instrument is exactly the mistake the commission's amendment
described, and it would have been available to make.

*(The commission's independently measured floor was ~5% at n=2; on my frame and
protocol it is 8.7%. Both say the same thing.)*

### 2.2 The paired form works, with a stated resolution

`pair19.sh` alternates A,B within a round (and B,A on even rounds, so a
"first run of a pair is colder" effect cannot masquerade as a delta) and reports
the per-round **paired ratio**. Common drift divides out. Null, n = 8, identical
source:

```
cycles/node      med=1.0009  min=0.9559  max=1.0436  spread=8.77pp
cache-miss/node  med=1.0112  min=0.9862  max=1.0659  spread=7.97pp
instr/node       med=1.0000  min=1.0000  max=1.0000  spread=0.00pp
```

Pairing removes the 8.7% systematic — the median comes back at 1.0009 — but the
round-to-round spread is 8.77pp.

**And it does not improve as √n, which §5.1 had to establish the hard way.** The
same null re-run at **n = 20** gives `med=1.0163 spread=19.17pp`: more rounds
moved the median *further* from unity. The interference drifts more slowly than
the measurement runs, so the rounds are not independent samples and averaging
does not average it out. **Treat ±2% as the floor for a paired cycles median on
this box at any affordable n**, and re-null at the SAME n as any arm you intend
to quote.

**What that means for this study, stated plainly:** the landed arms are −0.738%
of instructions. If IPC is flat that is a ~0.7% cycles effect — **below what this
instrument resolves at any round count I can afford.** So the cycles/cache
numbers in §5 are reported as a **direction with its spread**, never as a
magnitude, and they are not part of the accept/reject decision. `instructions:u`
is the gate; the second axis is a tripwire for the specific failure mode where an
arm trades instructions for stalls.

**The third leg of the instruction axis, worth recording**: `instr/node` came
back `med=1.0000 min=1.0000 max=1.0000 spread=0.00pp` across 16 interleaved
runs of two independently built binaries. That is the within-pair control, and it
is the same fact the commission established with two independent PGO profiles.

---

## 3. WHAT LANDED

All three arms are the same shape: **a fact the caller has already established,
re-derived at runtime by the callee.** None of them adds an instruction anywhere,
which is what makes them immune to g1's failure mode (§6.1).

### 3.1 g2 — `push_kept`, −28.46

`DeltaBuf::push` is the choke point every recording site flows through, and it
filters with `threat_pair_kept(attacker, victim)` — a `MAP[a][v]` load and a sign
test. Perf study §6-T2 put it there deliberately, and for two sites it is doing
real work.

There are **twelve** emission sites. For **eight** of them the filter is dead,
because the pair is a structural property of the site:

* `record_targets` iterates `KEPT_VICTIMS[attacker]`, so **every victim type it
  visits is kept by construction** — 1 site;
* the **five** pawn-push emitters are pawn→pawn, and `MAP[Pawn][Pawn] = 0`
  (`record_square_touch`'s outgoing and incoming pushes, `record_outgoing`'s, and
  `record_incoming_mutate`'s pair) — 5 sites;
* the two incoming scans in `record_square_touch` sit inside *hoisted*
  `threat_pair_kept(Knight | Pawn, p.piece_type)` guards — the same test,
  already evaluated once per touch — 2 sites.

`push_kept` is `push` without the filter. **The drop set is not duplicated:**
there is still exactly one definition (`MAP`, via `threat_pair_kept`), and
`push_kept` carries a `debug_assert!` that re-checks the caller's claim on every
event — which the whole `sota` suite runs under, including the per-node
`assert_matches_refresh` oracle. That is the fidelity property study 18 §6.1
wanted and could not get from its rejected a2.

The **four** sites that keep the filter are `ripple`'s two emissions (the
discovered victim can be a king; a bishop/rook→queen pair is dropped) and
`record_incoming_mutate`'s fan-loop pair (the fans carry rook/bishop/queen
attackers against an `old`/`new` victim that can be a queen).

Counted in the shipped tree: **8 `push_kept`, 4 `push`.**

### 3.2 g3 — the victim mask becomes a const parameter, −9.34

```rust
fn record_targets<const KEPT: u8>(…)
```

`KEPT_VICTIMS[attacker.piece_type as usize]` was a table load feeding **six
runtime bit tests** over `PieceType::ALL`, to re-derive a value the caller knows
at compile time: `record_square_touch`'s outgoing section reaches this from
inside a `match p.piece_type` arm, and `record_outgoing`'s pawn arm is literally
`PieceType::Pawn`. With the mask as a const parameter the scan disappears — the
body is emitted once per victim type actually in the set (3 for a pawn, 4 for a
bishop/rook, 5 for a knight/queen) with no test at all.

Six call sites are constant. The seventh, `record_outgoing`'s catch-all, is
spelled out as a two-way match: knight and queen share a `KEPT_VICTIMS` row
({P,N,B,R,Q}) and bishop and rook share the other ({P,N,B,R}), so it is **two**
specialisations, not four.

The idiom is the const-generic specialisation `movegen::generate::<GT>` already
uses. `KEPT` is a `u8` and not a `PieceType` because const generics cannot yet be
user enums — callers pass `KEPT_VICTIMS[…]`, so it is still the one `MAP`-derived
table, consulted by the compiler instead of the CPU. A `debug_assert_eq!` pins
the const against the attacker's actual row.

This arm is **−3.531 branches/node**, by far the largest branch move of the
three, which is the signature you want: the deleted work was branches.

### 3.3 g4 — a guard that is dead inside its own enclosing `if`, −7.61

```rust
if p.piece_type != PieceType::King {
    if threat_pair_kept(PieceType::Knight, p.piece_type) {   // ← always true
```

`MAP`'s knight row is `[0, 1, 2, 3, 4, -1]`: a knight threatens every victim type
**except a king**. The block is already inside `p.piece_type != King`. The inner
test could only ever be true — a `MAP` load and a branch per touch to re-derive
the condition of the enclosing `if`. It is now a `debug_assert!`.

**The pawn scan directly below it is NOT the same case and keeps its guard**, and
saying so is the point: `MAP`'s pawn row is `[0, 1, -1, 2, -1, -1]`, so a pawn
threatens only {pawn, knight, rook} and a bishop or queen victim really does skip
the whole scan. Two adjacent guards, one dead and one load-bearing, and the only
way to tell is to read the table.

Also in g4: `record_incoming_mutate`'s pawn-push pair, both pawn→pawn under their
own guards, moved to `push_kept`.

---

## 4. BOTH ENGINES IN ONE 30k FRAME — AND WHY THE FATTEST STAGE IS THE WRONG TARGET

Study 18's Stockfish column was measured at the **3k** budget. Study 17 §4.3 had
already warned that the ratio moves with the budget and instructed that the stage
split be run at 30k. Doing that to *our* side and comparing against a *3k*
Stockfish would have been the exact error study 18 §9.1 rule 3 names. So
Stockfish was re-profiled here at 30k, same 50 positions, same script:

* ARCHON **1,500,780** nodes, 15381.5 i/n total
* Stockfish **1,500,789** nodes, 7973.2 i/n total

Nine nodes apart. Self-cost, the form that sums to each engine's own total.

### 4.1 Resolution — 1.289× total, **1.076× per unit of work**

| | ARCHON | SF |
|---|---:|---:|
| symbol | `resolve_both` 285.65 + `resolve_into` 4.24 = **289.89** | `append_changed_indices` **224.89** |
| perspective-resolves / node | 0.5776 × 2 + 0.0406 = **1.1958** | **0.9984** |
| **i per perspective-resolve** | **242.4** | **225.3** |
| | | **1.076×** |

Ours resolves both perspectives in one pass over the event list (study 9 item 1),
sharing the two `sf_code` decodes; SF's `append_changed_indices` takes a
perspective and is called once per side. The arithmetic inside is the same
shape — I read `FullThreats::make_index` at
`eca43a97:src/nnue/features/full_threats.cpp:195-206` against our
`ThreatResolver::index_decoded` and they are the same three table lookups
(`lut1[att][atd][dir] + offsets[att][from_o] + lut2[att][from_o][to_o]`) on the
same oriented squares.

**So the +65.0 is not spelling. It is that we perform 1.198× more
perspective-resolves per node** — and that is a design difference, not a defect:

| | ARCHON | SF |
|---|---:|---:|
| from-scratch refreshes / node | **0.0204** | **0.1674** |
| refresh column | `refresh_threat_into_at` 41.10 + `refresh_psqt_threat_at` 15.37 + `discharge_refresh` 0.81 = **57.28** | `append_active_indices` **100.04** |
| **resolve + refresh, combined** | **347.17** | **324.93** |

A refresh **terminates the ancestor walk**, so refreshing more often means
folding fewer plies. SF's threat refresh triggers on **any own-king move**
(`HalfKAv2_hm::requires_refresh`, because its accumulator fuses HalfKA and
threats into one vector); ours triggers only on a king crossing the d/e mirror,
which study 18 §4.4 correctly identified as an architectural advantage —
Stockfish's own comment at `nnue_accumulator.cpp:112-113` concedes it.

**The advantage is real and it is smaller than it looks: 8.2× fewer refreshes
buys 22.2 i/n net, because the walks it lengthens cost most of it back.** That is
a correction to the natural reading of study 18 §4.4, and it is the reason
resolution should not be attacked: it is within 8% of Stockfish per unit of work,
and the unit count is a trade we are already winning, narrowly.

### 4.2 Where we are simply ahead, and where we simply pay

| | ARCHON | SF | |
|---|---:|---:|---|
| from-scratch refresh (§4.1) | 57.28 | 100.04 | **0.57× — we are faster** |
| push-time snapshot (`apply_at`) | **81.25** @ 1.0000/node | **0** | SF's `AccumulatorStack::push` (`nnue_accumulator.cpp:64-71`) does no accumulator work at all |

`apply_at`'s 81.25 has **no Stockfish counterpart**: we store an 80-byte
`ThreatPosition` plus the `DirtyPiece` on every made move so that fold-time
regeneration survives a sibling overwriting the live board; Stockfish instead
generates its `DirtyThreats` list eagerly inside `do_move`. It is the price of
laziness, paid on 100% of moves, and it is study 18 remainder item 7.

### 4.3 Generation — the block, at 1.30× and 1.52× per call

| | ARCHON | SF |
|---|---:|---:|
| | `record_square_touch` 203.02 | `update_piece_threats<true>` 220.42 |
| | `build_deltas` 81.32 *(from the split; carries the outlining tax)* | `update_piece_threats<false>` 63.06 |
| | `ripple` 37.44 | |
| | `record_targets` 25.57 | |
| | `record_outgoing` 20.18 | |
| **total** | **367.53** | **283.48** |
| | | **1.30×, +84.1** |
| generator calls / node | **1.0865** | **2.2190** |
| **i per generator call** | **186.9** | **122.6 / 149.9** |
| | | **1.52×** |

**We call the generator at 0.49× Stockfish's rate and pay 1.52× per call.** The
arithmetic that makes this the right target: at SF's per-call cost our column
would be `367.53 × 122.6/186.9 ≈ 241` — **below** SF's 283.48. The laziness is
already paid for; the per-call gap is what is left, and it is worth ~126 i/n in
this frame.

The landed arms all attack exactly this, and they are the first three bites of
it.

---

## 5. THE SECOND AXIS

### 5.1 On the landed tree — a direction, explicitly not a magnitude

Paired, interleaved, baseline vs `g4` (shipped), **20 rounds** (`pairs.txt`):

```
pair g4-dead-knight-guard / nullpair-A   n=20 rounds
  cycles/node      med=0.9794  min=0.9442  max=1.0369  spread= 9.27pp
  cache-miss/node  med=0.9652  min=0.9083  max=1.0483  spread=14.00pp
  instr/node       med=0.9926  min=0.9926  max=0.9926  spread= 0.00pp
```

The instruction control is exactly 0.9926 with **zero** spread across 40
interleaved runs, and 0.9926 is precisely the −0.738% the probe measured — the
two instruments agree to four decimals.

**And the cycles median is the trap.** −2.06% cycles against a −0.74% instruction
change is a lovely story ("we deleted stall-y work"), and study 18 §9.1's
companion rule — *when the story gets beautiful, attack it hardest* — applies
directly. My paired null had been run at **n = 8**, and a median measured at
n = 8 cannot bound a median measured at n = 20. So I re-ran the null at
**matching n = 20**, on two builds of the SHIPPED tree:

```
pair nullB-B / nullB-A                   n=20 rounds     <- IDENTICAL SOURCE
  cycles/node      med=1.0163  min=0.9384  max=1.1301  spread=19.17pp
  cache-miss/node  med=1.0219  min=0.9273  max=1.0852  spread=15.79pp
  instr/node       med=1.0000  min=1.0000  max=1.0000  spread= 0.00pp
```

**The null's own median is 1.63% from unity — comparable in size to the arm's
2.06% and in the OPPOSITE direction — with a spread twice the arm's.** The
−2.06% is not distinguishable from nothing. Same for cache misses: the arm reads
0.9652 against a null of 1.0219.

Note also that **going from n = 8 to n = 20 moved the null's median AWAY from
unity** (1.0009 → 1.0163). More rounds did not converge it, for the same reason
§2.1 gives: the interference drifts more slowly than the measurement runs, so
averaging does not average it out.

**So: I do not claim the −2.06%, and the null is why rather than caution being
why.** What I claim is the sign — cycles and cache misses both moved in the same
direction as instructions, and neither moved against them, which is the only
thing the tripwire was installed to detect.

The prior for a cache regression here was very low to begin with: every one of
g2/g3/g4 deletes a load from an L1-resident 36-byte table (`MAP`) or a branch,
and none moves, adds or reorders a single weight-row or accumulator access.

### 5.2 THE AXIS EARNS ITS KEEP SOMEWHERE ELSE — and it refutes this study's own hypothesis

The second axis is nearly useless for pricing a 0.7% arm on this box, and it is
*decisive* for ranking. Self-profile, my frame (5,001,562 nodes, wl50 @ 100k,
native AVX-512, thin LTO, non-PGO), `cache-misses:u`:

| symbol | share of ALL cache misses |
|---|---:|
| `avx512::i8_deltas_into` (threat fold) | **23.49%** |
| `avx512::i16_deltas_into` (HalfKA fold) | **23.14%** |
| `avx512::i16_deltas_dual` | **19.12%** |
| **the three accumulator-fold kernels** | **65.75%** |
| `at_full` | 4.69 |
| `MovePicker::init_quiets` | 3.23 |
| `SotaHead::forward_into` | 2.91 |
| `finny_refresh_into` | 1.27 |

**An earlier draft of §8 hypothesised that `at_full`'s stall time is the random
32-byte threat-PSQT row load inside `resolve_both`** — one random access per
event per perspective into a 1.94 MB table, ~12 per fold. It is the only large
random table in the stage, the story was clean, and **it is wrong**: `at_full`
owns **4.69%** of cache misses, not the bulk. The hypothesis is withdrawn, and it
was withdrawn by the axis the commission added, on its first real use.

The misses are in the **weight-column streaming inside the fold kernels** — ~6
random 1280-byte columns per perspective-ply out of a 77 MB threat table, plus
the HalfKA equivalent. §8 item 1 is rewritten around that.

---

## 6. WHAT DID NOT LAND

### 6.1 g1 / g1b — Stockfish's `occupiedNoK`, +4.63 and +1.82, and the number that explains both

`eca43a97:src/position.cpp:1195-1200` computes the discovered piece as
`ray & (rAttacks | bAttacks) & occupiedNoK`. Ours was
`ray_pass(sl, s) & r.occ & qatt`. Two differences, and both looked like free
wins:

* `qatt & occ` does not depend on the slider, so it looked like it belonged in
  `record_square_touch` rather than inside `ripple`; and
* **kings are excluded on SF's side and were not on ours** — so we discovered a
  king, resolved it through `ThreatPosition::piece_on` (King is that function's
  five-test *fallthrough*, i.e. its worst case), built a `ThreatDelta` and handed
  it to `push`, which dropped it, because every row of `MAP` has `-1` in the King
  column.

The event stream is provably identical either way. **g1 measured +4.63.** Split
into halves, **g1b (the king mask alone, no hoist) measured +1.82**, so the hoist
alone was **+2.81**.

The explanation is a call count, taken from the base profile's own annotation:

```
ripple, from the rook fan     462,160 calls
ripple, from the bishop fan   259,967 calls
                              -------
                              722,127  =  0.481 / node
record_square_touch                       1.0865 / node
```

**0.443 ripples per touch.** Anything hoisted from `ripple`'s body to
`record_square_touch`'s body is paid **2.26× more often**, and the whole x-ray
section is only 37.4 i/n inclusive to begin with. The king-exclusion saving is
real but tiny — branches fell 0.111/node, so the events exist — and it does not
pay for two instructions on a path that runs 2.26× more often.

**The rule this pays for, and it kills a family of arms rather than one:** in
this module the inner loop is *rarer* than the outer one. Loop-invariant code
motion is only a win when the invariant is used more than once, and here the
expected number of uses is **0.44**. Before hoisting anything out of `ripple`,
`record_targets` or either incoming scan, read the call count.

### 6.2 Not attempted, and why

* **Merging the two slider fans into SF's single `sliders` union loop.** It
  replaces two loop set-ups with one, but each element then needs
  `ThreatPosition::piece_on` (five type tests) instead of the fan-implied type
  plus `color_on` (one). Given §6.1's 0.443 elements per touch, the loop-set-up
  saving is paid on ~0.44 elements and the `piece_on` cost on all of them. Same
  shape as the arm that just failed twice.
* **Attacking `resolve_both` because it is the biggest stage.** §4.1: it is
  1.076× Stockfish per unit of work. That is what the matched frame is for.

---

## 7. THE PGO PREDICTION — TWO-SIDED, WITH THE KILL CONDITION NAMED

The calibration now runs in **both** directions, which it did not before study 18:

| study | class | probe → PGO |
|---|---|---|
| 12–15 | data layout, `memcpy` | probe **overstated** 1.12–2.0× |
| 17 | inlining boundaries (`#[cold]`) | probe **overstated** 2.17× (−114.5 → −52.8) |
| 18 | **work elimination** | probe **UNDERSTATED 2.94×** (−40.9 → −120.1) |

This study's arms are **work elimination and const specialisation** — the study-18
class, not the study-17 class. Nothing here is an `#[inline(never)]`, a `#[cold]`,
or a layout change; every arm deletes a table load or a branch outright.

**Predicted: −45 to −135 i/n in the PGO frame, most likely −60 to −100.**

**The falsifiable part is a floor, not the sign.** I predict
**|PGO| ≥ 0.8 × |probe|, i.e. not weaker than −36.** If the panel comes back at
−20 or better, the model "deleted redundant table loads and branches survive PGO"
is **wrong**, and the most likely mechanism is specific and testable: `push`'s
`threat_pair_kept` test is a **heavily biased branch** — most pairs *are* kept —
so a PGO'd baseline may already predict it perfectly and retire it at near-zero
cost, making g2's deletion worth much less than thin LTO says.

That mechanism makes a **decomposition prediction the panel can check**: if it is
right, **g3 survives PGO and g2 does not**, because g3 deletes a *load and a loop*
while g2 deletes a *biased branch*. Measuring g3 alone would separate them, and
the answer would generalise to the whole `#[inline]`-choke-point-filter class
rather than to this arm.

I do **not** predict the sign can flip. There is no spelling of "do not load the
table" that PGO turns into a cost.

**Secondary prediction:** branches/node falls proportionally *more* than
instructions (−0.98% vs −0.738%), so IPC should not fall. If the PGO panel shows
cycles moving *less* than instructions, that is consistent; if it shows cycles
moving *against* instructions, §5's low prior was wrong and the tripwire fired.

---

## 8. RANKED REMAINDER — handoff to study 20

**Closed by this study, with numbers:** study 17 item 1b's second half (the
`at_full` stage split — §0.2, four stages priced, outlining tax bounded at
+85.5); study 18 item 1's premise (resolution is the biggest stage **and** the
wrong target — §4.1); study 18 item 2 partially (`record_square_touch` is
attacked, −45.41 probe, and the generation gap is now sized at 1.52× per call in
a matched frame).

**Do not reopen:** everything on study 18's list, **plus** any arm that hoists
work out of `ripple`, `record_targets` or the incoming scans into
`record_square_touch` (§6.1 — the inner loop runs 0.443× per outer call), **plus**
`resolve_both`'s per-event arithmetic (§4.1 — 1.076× SF).

### 8.0 RANK ON OUR OWN PROFILE SHARE FIRST — the comparative method has a blind spot

Every ranking studies 11–19 produced came from block-by-block comparison against
Stockfish, and **that method structurally cannot see anything without a
Stockfish counterpart.** Two self-profiles — the owner's (`bench` set, depth 16,
7.86 M nodes) and mine (wl50 @ 100k, 5,001,562 nodes) — both say so.

| symbol | owner, d16 bench set | **mine, wl50 @100k** | callgrind i/n share (mine) |
|---|---:|---:|---:|
| `search::NonPv` | 19.90% | 6.63% | 4.81% |
| **`at_full`** | **18.56%** | **5.29%** | 3.10% |
| `avx512::i16_deltas_into` | 13.14% | 12.44% | 20.55% |
| **`avx512::i8_deltas_into`** | *(not in top 13)* | **13.01%** | 24.35% |
| `avx512::i16_deltas_dual` | *(not in top 13)* | 8.14% | 5.21% |
| `SotaHead::forward_into` | 8.77% | 7.73% | 0.78% |
| **`finny_refresh_into`** | **9.15%** | **1.30%** | **0.49%** |
| `MovePicker::init_quiets` | 6.35% | 5.63% | 3.65% |
| `record_square_touch` | 1.98% | 1.13% | 1.32% |

**Three things follow, and only the third is safe to rank on.**

**(a) A cycle share is as frame-bound as an i/n.** The two self-profiles disagree
by **3.5× on `at_full`**, **7.0× on `finny_refresh_into`** and 3.0× on
`search::NonPv`. That is not error — wl50 is an endgame-skewed calibration set
and the bench set at fixed depth is middlegame-heavy, so the piece count, the
feature count and the king-move rate all differ. Study 18 §9.1 rule 3 ("an
estimate carries its frame") applies to profile shares exactly as it does to
i/n. **Games are played into middlegames, so the d16 bench-set frame is the one
to rank on** — but every share below is stamped with which frame it came from.

**(b) Instruction share and cycle share are different rankings, and the gap IS
the information.** `finny_refresh_into` is **0.49% of instructions and 9.15% of
cycles** — an ~19× IPC penalty; `forward_into` is 0.78% vs 8.77%, ~11×. These are
stall-bound symbols that `instructions:u` is constitutionally blind to, which is
the commission's original point made concrete. Symmetrically, `i16_deltas_into`
is 20.55% of (AVX2-under-valgrind) instructions and 12–13% of cycles.

**(c) The frame-ROBUST fact, and it reframes the whole block.** Sum the
accumulator-maintenance symbols:

| | owner's frame | my frame |
|---|---:|---:|
| **accumulator maintenance** (`at_full` + the three fold kernels + `current_full` + `finny_refresh_into` + `push`) | **44.1%** | **45.4%** |
| **network head** (`forward_into`) | **8.77%** | **7.73%** |

**Two workloads that disagree 3–7× on individual symbols agree to 1.3 points on
the aggregate.** Keeping the accumulator current costs ~5× what using it costs.
**We are not slow because the net is wide.** Any future width argument has to be
made against this number, and it is the one thing in this section I would defend
without re-measuring.

### 8.1 The table

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **PREFETCH THE WEIGHT ROW AT INDEX-RESOLUTION TIME — Stockfish does, we do not.** The three fold kernels are **65.75% of all cache misses** and ~33.6% of cycles (§5.2, my frame). `append_changed_indices` takes `prefetchBase`/`prefetchStride` and issues `prefetch<READ, LOW>(prefetchBase + index * prefetchStride)` **immediately after computing each index** (`eca43a97:src/nnue/features/full_threats.cpp:295-299`) — it overlaps the row fetch with the rest of the resolve. Our phase split (`resolve_both` then the fold) is the *ideal* shape for this and leaves the latency fully exposed | **the largest single lever this study found** | **high on the mechanism; the size needs a timing frame** | **`compute`'s doc comment says a row-prefetch "MEASURED A LOSS (−2.5% NPS full-row, ~−0.4% first-two-lines)". Do not treat that as settled:** it is study-4 era, it predates the phase split, and the coordinator measured NPS instruments at ~24% (match-PGN) / ~1% (paired) noise on this box — **−0.4% is inside the noise of the instrument that rejected it.** Re-run it on a PGO binary with a paired timing protocol at n ≫ 20, and expect instructions to RISE |
| 2 | **`finny_refresh_into`: 0.49% of instructions, 9.15% of cycles (owner's frame) — a ~19× IPC penalty, and it is in NO study's ranked remainder.** Pure stall, invisible to every instrument studies 12–19 used | up to ~9% of cycles | high that it is stall-bound, unknown how much is recoverable | it is a Finny-cache diff over a wide accumulator; profile `cache-misses:u` *within* the symbol (`perf annotate`) before proposing anything. **Note it is 1.30% on my endgame-skewed frame — settle the frame first (§8.0a)** |
| 3 | **`MovePicker::init_quiets` — 6.35% of cycles (owner), 5.63% (mine), 3401.3 instructions per call, and STILL never split.** Study 18 item 8 ranked it "low confidence"; at 6.35% of cycles that ranking is wrong | 0–40 i/n, but 6% of cycles | medium | `cg19.sh` + the `split19.py` treatment. **Split before outlining anything** — that is the lesson §0.2 just paid for on `at_full` |
| 4 | **Generation's remaining per-call gap: 186.9 vs 122.6 i/call.** g2/g3/g4 took the redundant-guard family; what is left is structural. Two named divergences: SF's incoming scan accumulates **one** `incoming_threats` bitboard and walks it **once** with a mailbox `piece_on`, where we run three or four typed loops; and SF's `PawnPushOrAttacks[c][s]` is a **precomputed table** where our pawn arm does `Square::shifted` + a `contains` test (up to three per pawn touch). `core/src/option.rs` attributes **19.7 i/n** inside `record_square_touch` alone | **~126 i/n callgrind at parity** | **high — matched frame, both engines** | the `PawnPushOrAttacks` table first: it is a pure deletion of `Option` machinery, `pawn_push_or_attacks` already exists in `sota_features.rs` (it is just computed, not tabulated), and it does not hoist across the §6.1 frequency boundary |
| 5 | **`ThreatPosition::piece_on` is five type tests where `Board`'s is one mailbox byte** — study 18 §3.2 measured that as the +7.1 toll of a1a and it is now load-bearing for every "adopt SF's shape" arm, because SF's shape is *union-mask + `piece_on`*. Adding a 64-byte `[u8; 64]` mailbox to the snapshot makes `toggle` an XOR on one byte (0 = empty, so XOR is still its own inverse) and `piece_on` a load | unlocks item 1 and §6.2 | medium | `ThreatEntry` grows 5,248 → 5,312 B (+1.2%); `from_board` gains one 64-byte copy at 1.0/node. Measure `from_board` alone first — it sits inside `apply_at`'s 81.25 |
| 6 | **`apply_at` is 81.25 i/n at 1.0000 calls/node with NO Stockfish counterpart** (§4.2). `ThreatPosition::from_board` unions six type boards across colours because `Board.pieces` is `[[Bitboard; 6]; 2]` (colour-major). A `by_type: [Bitboard; 6]` maintained in `Board` would make `from_board` a straight 80-byte copy | 20–40 callgrind | medium | it touches `put_piece`/`remove_piece`/`move_piece`, so it is a **D24 perft-ritual** change; +2 ops on ~1.5 placements/move against −6 ORs and −6 loads per move |
| 7 | ~~The threat PSQT row is a random 32-byte load from a 1.94 MB table, per event, per perspective — the one place in the threat block with a genuine cache-miss story.~~ **WITHDRAWN BY §5.2 BEFORE IT SHIPPED.** `at_full` owns **4.69%** of cache misses, not the bulk; the misses are in the fold kernels. It was a clean mechanism, an attractive story, and false — recorded here so study 20 does not re-derive it | — | refuted | — |
| 8 | **The `acc fold` block is still unreadable here and is still the biggest number in the table** (`i16_deltas_into` 3161.01 + `i8_deltas_into` 3745.34 + `i16_deltas_dual` 802.16 = 7708.5 vs SF's fused `apply_combined` 1489.93). Study 18 §7's runtime-dispatch trap. One thing the matched frame *does* show and is worth writing down: **SF fuses the HalfKA and threat accumulator updates into ONE pass over the output vector; we make two** | unknown | high on the shape, **zero on the number** | needs a COMPILE-TIME AVX2 build of ours that forces the dispatch instead of reading CPUID — study 18 item 5's open half, unchanged |
| 9 | Inherited and unchanged from study 18: item 3 (`ripple`'s cancelling pairs, capped at 1.034 overfold), item 4 (the six-argument re-try of a2), item 6 (SF's shifted-bitboard pawn enumeration), item 9's whole inherited list | | | |

### 8.1 Three method rules this study hands over

1. **THE FATTEST STAGE IS NOT AUTOMATICALLY THE TARGET — GET THE REFERENCE INTO
   THE SAME FRAME FIRST.** §4.1. `resolve_both` is 51% of the split and 1.076× of
   Stockfish per unit of work. A study that split `at_full` and then attacked the
   biggest number would have spent its whole budget against parity. The split
   tells you *where the cost is*; only the matched-frame reference tells you
   *whether it is a defect*.
2. **BEFORE HOISTING, READ THE CALL COUNT — THE INNER LOOP MAY BE RARER THAN THE
   OUTER ONE.** §6.1. `ripple` runs 0.443 times per `record_square_touch`, so
   loop-invariant code motion out of it is a 2.26× *penalty*, and it cost this
   study two arms to learn on a change that provably removes work.
3. **A CONTENDED METRIC NEEDS A PAIRED INSTRUMENT, THE PAIRING MUST BE NULLED
   TOO, AND THE NULL MUST BE AT THE ARM'S OWN n.** §2, §5.1. Sequential cycles
   moved 8.7% between bit-identical builds; pairing removed the bias but left a
   spread no affordable round count beats. The part that nearly caught me: my
   null was n = 8 and my arm was n = 20, and at n = 20 the null median is
   **1.0163** — *further* from unity than at n = 8, and comparable in size to the
   arm's own displacement. **A null at a different n is not a null.**
4. **RANK ON YOUR OWN PROFILE SHARE FIRST; USE THE REFERENCE ONLY TO SAY HOW
   MUCH OF IT IS RECOVERABLE.** §8.0. Every ranking studies 11–19 produced came
   from diffing blocks against Stockfish, and that method cannot see a symbol
   with no Stockfish counterpart. `finny_refresh_into` is 9.15% of cycles and has
   appeared in no ranked remainder; `record_square_touch`, which has a clean
   counterpart, was ranked second and is 1.98%. **Work flows to what is
   comparable rather than to what is expensive** unless the ranking starts from a
   self-profile.
5. **A PROFILE SHARE CARRIES ITS FRAME AS HARD AS AN i/n DOES.** §8.0a. Two
   self-profiles of the same binary disagree **3.5× on `at_full`** and **7.0× on
   `finny_refresh_into`** purely because one workload is endgame-skewed and the
   other middlegame-heavy — while agreeing to 1.3 points on the
   accumulator-maintenance aggregate. Rank on the aggregate, or name the frame.

---

## 9. WHAT IS OWED BEFORE THIS MERGES

Green **here**, on HEAD:

| check | result |
|---|---|
| `cargo test` (default; includes the fast perft tier + checked mini-tier) | **608 passed, 0 failed** |
| `cargo test --features sota,arch-d48x` | **695 passed, 0 failed, 8 ignored** |
| `cargo clippy --all-targets` | **0 warnings** |
| `cargo clippy --all-targets --features sota,arch-d48x` | **6 warnings, all pre-existing** — 5 in `simple_halfka.rs`, 1 in `protected_halfka.rs`, none in a file this study touches (same set and count study 18 recorded) |
| `cargo fmt --check` | **clean** |
| `cargo test --release --test perft_suite -- --ignored perft_full_suite` (**D24**) | see §9.1 |
| probe-tree node identity (5,001,562) + eval hash `5ca978e012da1bb0`, every arm | **PASS, all ten lines** |

Deferred to the owner, in this order:

1. **`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=4 bash tools/check.sh` — MANDATORY.**
   This study is entirely under `src/eval/nnue/`, which the `Bench:` pin is blind
   to. Expect **all three pins UNCHANGED**: `Bench: 587683`, eval hash
   `5ca978e0…8b38`, d13 nodes `1922947`. Every arm is node-identical and
   eval-hash-identical, so **a moved pin is a bug in this study, not a new pin.**
2. **`bash tools/build_pgo.sh sota,arch-d48x engines/bin/archon-s19
   engines/canonical/net.cenn`** — note study 18's recorded trap: `build_pgo.sh`
   prepends `engines/bin/` to its second argument, so an absolute path fails the
   copy after a successful build. Then the two-arm panel against
   `engines/bin/stockfish-eca43a97`, which is the test of §7's **−45 … −135**
   prediction and its named kill condition.
3. **If §7's floor is breached**, run g3 alone (revert g2 and g4, keep the const
   parameter) — that is the decomposition experiment that separates
   "deleted load" from "deleted biased branch", and it is one probe cycle.

### 9.1 Node identity, argued rather than asserted

Two independent statements covering different halves, following study 18 §10.1:

* **The wl50 eval hash** (`5ca978e012da1bb0`, unchanged on all ten probe lines)
  is a from-scratch forward pass on 50 positions. It pins the **enumeration** —
  and g3 changed `record_targets`, which the from-scratch path reaches through
  `record_outgoing`. It pins nothing about `compute`.
* **The node count** (5,001,562, unchanged on all ten lines including both
  rejected arms) is a 5 M-node search. Any divergence in the incremental
  accumulator changes a score, changes a cutoff and changes the tree. It pins the
  **incremental generator**, which is what g2/g3/g4 actually edit.

Plus the `cfg(debug_assertions)` oracle (`SotaContext::assert_matches_refresh`)
comparing incremental against from-scratch at **every** evaluated node — which
the entire `sota` debug suite runs under, and which is where g2's and g4's new
`debug_assert!`s live. Those assertions are the reason the "the caller already
proved it" argument is checkable rather than merely stated: **if any of the six
sites could emit a dropped pair, the 695-test sota run would abort on it.**

---

## 10. THREATS TO VALIDITY

1. **Nothing here is PGO-measured.** §7 is a prediction, not a result. The GPU
   trainer held the box all session and `lto="fat"` was not attempted.
2. **THE CYCLES AND CACHE-MISS AXES CANNOT RESOLVE THIS STUDY'S ARMS.** §2.2,
   §5.1. At the arm's own n = 20 the paired null median is **1.0163** with a
   19.17pp spread, against an arm median of 0.9794. I report a direction and
   explicitly decline to claim a magnitude.
3. **I RAN THE n = 20 NULL AFTER THE ARM, NOT BEFORE IT.** §5.1. Rule 1 says null
   first, and I nulled the paired instrument at n = 8 and then measured the arm
   at n = 20 — which is a different instrument configuration and therefore an
   un-nulled one. I noticed because the arm's median looked too good, not because
   the protocol caught it. Had the arm's cycles median landed at a *plausible*
   −0.8% instead of a suspicious −2.06%, I would probably have published it. **The
   n = 20 null says a −0.8% median would have been indistinguishable from
   nothing too.**
4. **I destroyed my own first n = 20 null run** by deleting the stashed binaries
   in `bin/` while `pair19.sh` was still executing against them. The rerun is the
   one quoted, and it is on two builds of the *shipped* tree rather than the
   baseline — an equally valid null (identical source, two builds) but not the
   same pair as the arm's A side. Noted rather than smoothed over.
5. **The §0.2 stage split carries an outlining tax.** The whole-binary total moved
   +85.5 (+0.56%), and `fold_one` takes seven arguments — over the SysV
   six-register limit study 17 §5.1c priced at 14.2. The per-stage numbers are
   therefore upper bounds on each stage's inlined cost, jointly bounded by the
   +85.5. The **ranking** between stages is what should be carried forward, not
   the absolute values.
6. **Every i/n in §4 and §8 is a CALLGRIND-frame number and none of them is
   headroom.** Study 18 §5.3's rule: the frames disagree by up to 26× on a single
   arm, and there is no scalar that converts between them. The probe frame is the
   only one that carries the landed deltas.
7. **`build_deltas`'s 81.32 in §4.3's generation column comes from the SPLIT
   build**, because it has no symbol in the base build (it is inlined into
   `at_full`). It therefore carries part of the outlining tax, so the ARCHON
   generation total 367.53 — and hence the 1.30× — is **conservative in
   Stockfish's favour by up to ~80**. The per-call comparison (186.9 vs 122.6),
   which is the one item 1 is ranked on, uses only base-build symbols and is
   unaffected.
8. **SF's 283.48 generation column excludes `remove_piece` (14.15) and
   `put_piece` (5.67)**, which do threat-list orchestration *and* placement work
   that cannot be separated by symbol. Including them would move SF to 303.30 and
   the ratio to 1.21×. Both forms are printed here; the exclusion runs in
   Stockfish's favour.
9. **The "caller already proved it" argument for g2 is a source-level argument
   checked by `debug_assert!`, not a type-level guarantee.** A future recording
   site added with `push_kept` and a wrong pair would be caught by every debug
   test but not by the compiler. §9.1 states what the check actually covers.
10. **The 30k callgrind budget is not the probe budget.** Study 17 §4.3 measured
   `at_full` at 466.19 (3k) → 343.51 (30k) on the *pre*-study-18 tree; on this
   tree it reads 470.82 (3k, study 18) → 477.45 (30k). The eval rate barely moved
   between the two budgets here (0.4151 → 0.4315 calls/node), so that earlier
   drop does not reproduce on the current code. I did not chase why; it does not
   affect any comparison in §4, which is single-frame throughout.
11. **Box state**: a GPU training run held ~5.8 GB and 145% CPU for the entire
   session. Everything quoted rides on `instructions:u` over a fixed-node tree or
   on deterministic callgrind counts, both contention-immune — except §5, which
   says so.

## 11. ARTIFACTS (`perf-artifacts-19/`)

* `probes-fast.txt` — all ten probe lines with instruction count, i/n to four
  decimals, branches, node count and eval hash: three nulls, two rejected arms,
  three landed arms.
* `pairs.txt` — the paired second-axis runs, including the instrument's own null.
* `cg/summary-base30k.txt`, `cg/summary-split30k.txt` — the stage split, before
  and after, same frame.
* `cg/summary-sf30k.txt` — **Stockfish in the same 30k frame.** This is the
  artifact that turns §4 from an assertion into a measurement.
* `cg/annot-base30k.txt` — per-source-line attribution. The `ripple` call counts
  that kill §6.1's family of arms are in it, at the two loop bodies.
* `cg/callgrind-*.out` (+ `.log`) — the raw callgrind files.
* `scripts/probe19.sh`, `pair19.sh` (new), `nullpair19.sh` (new), `split19.py`
  (new), `cg19.sh`, `cgsum19.py`, `cgbuild19.sh`, `blocks19.py`, `dis19.sh`.

The Stockfish `noinline` build tree is still at `/home/nick/archon-sf17-eca43a97`
(study 17 left it, study 18 patched it); this study rebuilt nothing there. The
ARCHON AVX2 trees are `target-cg19/` (base) and `target-cg19s/` (split).
