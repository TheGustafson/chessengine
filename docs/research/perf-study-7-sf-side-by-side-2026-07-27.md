# ARCHON perf study #7 — the SF side-by-side: where the per-second gap actually is

Date: 2026-07-27 (morning). Seventh commissioned study in the chain (1:
`threat-channel-perf-study.md`, 2: `perf-study-2-broad.md`, 3:
`perf-study-3-beyond.md`, 4–6: `perf-study-{4,5,6}-*.md`). Mandate: instruction-
and micro-architecture-level ground truth of ARCHON-per-second vs Stockfish,
with side-by-side assembly of the hot kernels; plus a commissioned add-on
mid-study: quantify SF's dual-net (big/small) architecture and re-state the
comparison in light of it. Raw artifacts: `perf-artifacts-7/`.

**Study only — the repository working tree was never modified.** Two binaries
were built into the session scratchpad (an ARCHON v3 twin for callgrind, and
SF18 from the reference clone's `sf_18` tag at `ARCH=x86-64-avx2` for exact
call counts); both are measurement instruments, not changes.

> ## ⛔ CORRECTION 2026-07-27 — §7 item 2's framing is WRONG (and §3.2 already said so)
>
> The follow-on implementation agent (branch `perf-impl-1-threats`; its
> `docs/research/perf-impl-1-plan.md` merges with that work) re-measured the
> item-2 premises and refuted two of them. Both refutations are measurements,
> not opinions, and they are **accepted**:
>
> **1. There is no 3× row surplus; our threat fold is already near-minimal.**
> A new instrument computes, per move, the true minimal delta — the symmetric
> difference of the from-scratch active threat feature sets before and after —
> and compares it against the rows our fold actually touches: 5.99 resolved
> vs 5.80 minimal per perspective-ply, an **overfold factor of 1.034** over
> 76,651 perspective-plies. Any correct incremental scheme must fold
> essentially the rows we already fold.
>
> **2. "`at_full` is 15.7% of deep cycles" conflates the fold with the replay.**
> PGO inlines the fold kernel into that symbol — which this study's own §3.1
> shows, since `at_full`'s top opcodes (`vpmovsxbw` 21.2, `vpaddw` 9.1,
> `vpsubw` 8.6) *are* the fold. **§3.2 of this document already gives the
> honest split** (`at_full` self 638 Ir/node against `i8_deltas_into` 2,354),
> so §7 item 2 contradicts §3.2 and §3.2 is the correct one. Confirmed
> directly by putting `#[inline(never)]` on `apply_i8_deltas_into` and
> re-profiling: fold 6.28% of instructions, `at_full` plumbing 6.18%, of which
> ~1.3pp is the PSQ1 skip-channel fold and not replay either.
>
> **Net effect on the recommendation:** the addressable replay surface is
> **≈10.0% of instructions**, not 15.7% of cycles, and the fold (~7.6pp with
> PSQ1) is irreducible in *any* scheme. Item 2's "+5–8% NPS" target is
> therefore over-promised and should not be used for planning. Treat the
> ranked table in §7 as superseded for item 2 until the implementation agent's
> measured result lands. Items 1 and 3 are untouched by this correction.
>
> The error is the study's, not the reader's: it had the right number in §3.2
> and quoted the wrong one in §7.

**Source state measured (all pinned, per user instruction):** engine
`engines/bin/archon-d47v-pgo` (sha256 `57531e1bb53979dd…`, PGO+native,
`sota,arch-d47v` = 768/32/32) with net
`data/nnue/nets/screens/d47v-a4b2-cal400.cenn` (`6afeebaa15e7df37…`).
Its source commit is **`d1c9aa2`**, established by rebuilding a plain
`x86-64-v3` binary from `git archive d1c9aa2` and reproducing the PGO binary's
wl50/d13 tree **exactly** (total 2,118,123, all 50 per-position counts
identical). Opponents: `engines/bin/stockfish18` (official, x86-64-vnni512,
`adc2a111…`) and `engines/bin/stockfish16` (official, avx2/bmi2, `7013d6fd…`).
New identity pins for this binary+net: 800-FEN eval hash `dd08810005c92f9e…`,
wl50/d13 total **2,118,123**.

**Workloads:** wl50 = `awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens`
(endgame-heavy); `default` = the frozen 50-FEN SF bench list, which ARCHON's
`bench` reproduces verbatim from SF's `benchmark.cpp` — so `bench 16 1 N
default nodes` is a **byte-identical position set on all three engines**;
plus a 2-FEN deep arm (startpos + the `r1bq1r1k…` middlegame FEN) to
reproduce last night's movetime instrument.

**Box state: NOT quiet, the whole session.** `archon-d48x` GPU trainer (97%
GPU, ~1.3 cores), 4-worker SPSA campaign + fastchess (~4 floating cores),
load 5.2–5.9, <1 GB free RAM at one point (mid-study resource directive:
no rebuilds, nice 19, modest perf buffers — followed). Consequences: every
wall-clock/NPS/cycles number comes from interleaved same-round pinned
(`taskset -c 4`, nice 19) rounds with per-round ratios, median quoted;
**`instructions:u` is the load-immune instrument that carries the
conclusions** — it reproduced to zero spread across rounds in every arm.

---

## 0. HEADLINE

**(1) The standing differential is re-established and it barely moved: we are
1.31× SF18 on cycles/node** (study 3: 1.28×; the difference is within the box
band) **and 1.497× on instructions/node (deterministic)** on the matched
15M-node wl50 workload. Against SF16 we are **1.99× cycles / 1.84×
instructions**. The matched-bench NPS deficit vs SF18 is **1.33×, not the
1.5× in last night's log** — but both numbers are real: the deficit **grows
with search depth** (deep single-position arm: 1.53× instructions, ~1.40×
cycles, movetime-3s 1.44×). Last night's 1.5× is the deep-middlegame end of a
1.31–1.45× band, measured with the movetime instrument; the matched
fixed-node figure is 1.31–1.33×.

**(2) "SF18 has no threat channel" — the standing claim from last night's
synthesis — is WRONG, and correcting it reframes the whole threat debate.**
SF18's big net input is HalfKAv2_hm (22,528 dims) **plus FullThreats (79,856
dims — larger than our 60,720)**; `use_smallnet` in `evaluate.cpp` and
`UseThreats == (dims == Big)` in `nnue_feature_transformer.h` (sf_18 tag) are
the governing lines, and `update_accumulator_incremental<FullThreats,1024>`
is 11.3% of SF18's measured instructions. Stockfish pays for threats too —
what they have that we don't is (a) a **threat-free 128-wide small net** that
handles 14–24% of evals, and (b) a **~2× leaner threat implementation**:
~1,725 Ir/node of threat maintenance vs our ~3,475 on matched AVX2 paths, at
1.33× our accumulator width. Their scheme is strictly incremental along the
search path; ours pays a per-eval reconstructive replay (`at_full`, 15.7% of
deep cycles).

**(3) The 1.78× threat-cost figure is REFUTED; the honest grid is 1.25–1.39×
instructions (deterministic), 1.27–1.6× cycles.** Measured by stripping the
threat tensors from a quant checkpoint and packing full/stripped arms from
the *same* checkpoint, interleaved fixed-node rounds, node counts verified:

| arch | mix | instr full/halfka | cycles (band) |
|---|---|---:|---:|
| 768 (V, the study engine) | wl50 | **1.247** | 1.27 (1.18–1.44) |
| 768 | default/middlegame | **1.301** | ~1.53 (1.35–1.56) |
| 1280 (X, `archon-d48x`) | wl50 | **1.317** | 1.39 (1.385–1.401) |
| 1280 | default/middlegame | **1.389** | ~1.55 (1.48–1.63) |

At 120 Elo/doubling the channel is worth **~45–70 Elo of speed** depending on
mix and width — real and still the largest single item on the board, but not
last night's "+100 Elo / half our speed". **And a threat-free ARCHON does NOT
reach SF18's speed:** the stripped arm still measures **1.21× SF18
cycles/node** (5,270 vs 4,331 i/n; same interleaved rounds).

**(4) The dual-net question (commissioned add-on): measured exactly, and it
explains little of the gap vs SF18 — but almost all of SF16's speed.** By
callgrind call counts (AVX2 build proven node-identical to the official
binary):

- SF18 small-net evals: **14.4%** of evals on wl50, **24.0%** on default
  FENs; small subtree ≈ **6× cheaper** per eval (1,060–1,167 vs 6,626–6,859
  Ir). If those evals had run the big net, SF18's instruction total rises
  only **+3.65% / +6.50%**. Re-stated apples-to-apples ("SF forced big-always"),
  our instruction ratio would drop 1.497→1.44 (wl50) and 1.464→1.375
  (default) — a visible slice, not the story.
- **SF16 is a different animal: 82.0% (wl50) / 73.7% (default) of its evals
  run the CLASSICAL handcrafted eval**, not its net (`|psq| > 2048` tier, no
  NNUE re-entry — sf_16 `evaluate.cpp:1059`). Its 2,302 c/n is therefore not
  an NNUE-engine number at all and must stop being quoted as a net-engine
  speed target. SF16-vs-SF18 itself is instructive: Stockfish accepted
  **1.48× more cycles/node** going 16→18 (2,302→3,401) for eval quality.

**(5) The remaining deficit is uniform, not concentrated — and it is
instruction COUNT and code SIZE, not execution quality.** At deep search both
engines spend ~54% of cycles in the eval layer; our eval block is 1.40× their
cycles, our shell 1.39×. Whole-engine counters (matched 15M-node trees):

- **we win**: IPC 1.41 vs 1.27; branch-misses/node 0.94×; DRAM fills 0.82×;
  L3 fills 0.85×
- **we lose**: instructions/node **1.50×**, branches/node **1.99×**, L1d
  loads 1.34×, L1d misses 1.38×, L1←L2 fills 1.44×, **icache misses 2.27×**

The machine executes our code *better* than SF's per instruction; we simply
issue half again as many instructions per node from hot functions that are
2–3× the static size of Stockfish's (PGO-grown: `next_move` 3,199 static
instrs vs SF18's 1,763; `search::<NonPv>` 5,168; `at_full` 2,287).

**(6) The assembly side-by-side found one shared tax and two design gaps.**
Shared: both engines' #1 eval opcode is `vpmovsxbw` (~21% of our `at_full`,
~20.5% of SF18's `evaluate`) — the i8→i16 widen of threat weights; not a
differential, and a quant-format rider would pay both sides. Design gap 1:
our threat resolution is a replay (at_full walks + 8-slot override scan +
PSQ1 row loads) where SF18's is pure incremental hops — this is the single
largest structural difference in the eval layer. Design gap 2: our shell is
2× as branchy (783 vs 395 branches/node) and `next_move`+`search` carry
2.27× the icache misses — the cost of PGO-inflated monolithic functions.

---

## 1. GROUND TRUTH — matched fixed-node counters

`bench 16 1 300000 <fens> nodes`, all three engines, one run of each per
round, `taskset -c 4`, nice 19; 6 rounds (G1) + 3+3 (cache groups). Node
totals: ARCHON 15,001,563 / SF18 15,010,238 / SF16 15,008,968 (0.06% apart).
Loaders included (<1% at this node count). Full parse:
`perf-artifacts-7/counters-matched-fixednode.txt`.

| per node | ARCHON | SF18 | SF16 | A/SF18 | A/SF16 |
|---|---:|---:|---:|---:|---:|
| instructions (deterministic) | **6,482.8** | 4,330.6 | 3,532.5 | **1.497×** | 1.835× |
| cycles (median) | **4,588** | 3,401 | 2,302 | **1.314×** | 1.988× |
| IPC | 1.41 | 1.27 | 1.53 | 1.11× | 0.92× |
| branches | 783.5 | 394.7 | 344.8 | 1.985× | 2.272× |
| branch-misses | 24.7 | 26.4 | 16.9 | **0.937×** | 1.465× |
| L1d loads | 3,171 | 2,365 | — | 1.342× | 2.013× |
| L1d misses | 587.5 | 425.3 | — | 1.384× | 2.274× |
| L1←L2 fills | 564.2 | 391.4 | — | 1.440× | 2.326× |
| L3 (CCX) fills | 43.1 | 50.9 | — | **0.846×** | 2.097× |
| DRAM fills | 6.1 | 8.4 | — | **0.815×** | 1.989× |
| icache misses | 195.9 | 85.2 | 70.9 | **2.271×** | 2.763× |
| NPS (same-round ratio) | 1.125M | 1.558M | 2.264M | 0.751 | 0.497 |

The middlegame arm (default FENs, 5 rounds) is structurally identical:
instructions 1.464×, cycles 1.305×, NPS 0.756 — **the position mix does not
move the matched-bench differential.** The counter *shape* reproduces study
3's exactly (surplus at L2/L3/DRAM, deficit at L1/L0i), cross-validating both
sessions: SF18's deterministic rows reproduce study 3 to four digits
(branches/node 394.7 vs 394.8, L1d loads 2,365 vs 2,368).

### 1.1 Depth dependence — where "1.5×" lives

| instrument (same two positions) | instr A/S18 | cycles A/S18 | NPS S18/A |
|---|---:|---:|---:|
| 50-FEN benches, 300k nodes/pos (above) | 1.46–1.50 | 1.31 | 1.32–1.33 |
| startpos+midgame, 10M nodes/pos, 3 rounds | **1.530** (det) | 1.40 (1.31–1.45) | 1.32–1.47 |
| movetime 3s, same positions, Hash 64, 3 rounds | — | — | **1.44** (1.41–1.48) |

Last night's 749k-vs-1112k (1.48×) reproduces under its own instrument
(movetime singles: A 813–848k, SF18 1195–1211k here) while the matched bench
says 1.31–1.33×. The gap between the instruments is a real depth effect, not
noise: deep trees raise *our* per-node cost faster than SF18's — our
absolute i/n goes 6,483→9,401 (+45%) from the bench to the deep arm while
SF18 goes 4,331→6,145 (+42%), and the eval-stack items grow their share
(finny_refresh 7.8→10.8% of cycles, at_full 15.0→15.7%, current_full
8.7→8.9% from d14 to the deep profile). A 1.5M-node version of the same arm
is contaminated by startup (SF's LEB128 net decode) and *understates* the
ratio (1.19–1.24 cycles) — recorded so nobody re-derives it:
`perf-artifacts-7/movetime-reconciliation.txt`.

### 1.2 Search-quality context (not re-derived, consistent)

At depth 14 on wl50 the trees are: ARCHON 3.07M nodes, SF18 3.20M, SF16
4.71M — we reach equal depth on fewer nodes than either, consistent with last
night's EBF finding (1.40 vs 1.51). The deficit is throughput, as
established; nothing in this study moves that conclusion.

---

## 2. THE FLAT PROFILES, SIDE BY SIDE

Deep wl50 profiles (ARCHON 10M nodes/pos ×30 positions, 1.19M samples; SF18
6M×50, 358k samples; SF16 6M×50). Full tables: `perf-artifacts-7/flat-*.txt`.

| ARCHON (PGO; kernels inlined) | % | SF18 | % | SF16 | % |
|---|---:|---|---:|---|---:|
| `ThreatContext::at_full` | **15.70** | `Network<1024>::evaluate` | **34.36** | `Eval::evaluate` (NNUE inl.) | 25.60 |
| `Worker::search::<NonPv>` | 14.69 | `search<0>` | 16.07 | `do_move` | 11.20 |
| `SotaHead::forward_into` | 12.84 | `next_move` | 13.06 | `Evaluation::value` (HCE) | 9.31 |
| `MovePicker::next_move` | 11.07 | `refresh_cache<1024>` | 7.95 | `next_move` | 6.88+2.69 |
| `finny_refresh_into` | 10.83 | `do_move` | 5.39 | `search<0>` clones | ~15.2 |
| `current_full` | 8.91 | `Network<128>::evaluate` | 3.18 | `update_acc_refresh` | 5.66 |
| `make_move_with` | 5.73 | `qsearch<0>` | 2.12 | `see_ge` | 2.43 |
| `do_move` | 4.20 | `see_ge` | 2.04 | `MovePicker::score` | 2.65 |
| `record_square_touch` | 3.77 | `update_piece_threats` | 3.07 | | |
| `qsearch::<NonPv>` | 2.94 | `refresh_cache<128>` + upd | ~2.5 | | |

Block totals (deep): ARCHON eval-layer ≈ **54.4%** (threat 20.4 + HalfKA 19.7
+ head 14.2), shell ≈ 45.6%. SF18 eval-layer ≈ **54.3%** (big 34.4 + small
3.2 + refresh 9.3 + folds 2.5 + threat events 4.9), shell ≈ 45.7%. **The
shares are equal; the absolute costs are uniformly ~1.4× — there is no
concentrated item left**, which is what study 3 predicted ("no remaining
afternoon fix worth 16%") now confirmed against both opponents.

---

## 3. ASSEMBLY SIDE BY SIDE

Method: `perf script -F ip,sym,symoff` histogram × `objdump -d -C`, dynamic
(sample-weighted) + static mixes, register-width classes, load/store
classification (`perf-artifacts-7/scripts/opc7.py`); excerpts under
`perf-artifacts-7/asm/`. ARCHON numbers from the 1.19M-sample profile; SF18
from a 358k-sample profile (its `evaluate` = 123k samples).

### 3.1 The eval flagships: our `at_full` vs SF18's `evaluate`

| | ARCHON `at_full` (15.7%, 187k samples) | SF18 `Network<1024>::evaluate` (34.4%, 123k samples) |
|---|---|---|
| static instrs | 2,287 | 1,909 |
| top opcodes | **`vpmovsxbw` 21.2**, `vmovdqu` 11.2, `lea` 9.5, `vpaddw` 9.1, `mov` 8.7, `vpsubw` 8.6 | **`vpmovsxbw` 20.6**, `vmovdqa64` 18.0, `vpsubw` 13.8, `cmp` 8.3, `vpdpbusd` 6.0, `vpaddw` 4.0 |
| dyn width | zmm 45.1 / xmm 13.1 / gpr 41.7 | zmm 67.8 / gpr 30.6 |
| dyn mem | load 30.4 / **store 20.6** | load 58.2 / store 10.8 |

Readings: (a) the i8→i16 widen is the top op on **both** sides — a shared
format tax, only removable by an i16-weight or unified-quant format rev that
would cost table bytes; not a differential. (b) Our 20.6% dynamic stores are
the replay signature: `at_full` *writes back* a reconstructed accumulator
every eval, where SF18's evaluate reads incrementally-maintained state
(their store share is half ours and their zmm share correspondingly higher).
(c) Our 41.7% GPR half is the walk/override-scan bookkeeping SF simply does
not have inside eval.

### 3.2 Threat maintenance, per node (matched AVX2 callgrind paths)

| | ARCHON (569,208 nodes) | SF18 (511,221 nodes) |
|---|---:|---:|
| event generation | `record_square_touch` 323 Ir/n + `apply_at` 159 | `update_piece_threats` 270 Ir/n (incl. swap) |
| fold/update | `i8_deltas_into` 2,354 Ir/n | `upd_incr<FullThreats,1024>` 1,070 Ir/n |
| per-eval resolve | `at_full` 638 Ir/n | — (none; no replay stage exists) |
| **total** | **≈3,475 Ir/n** | **≈1,725 Ir/n** (+ threat share of `evaluate`) |

Their accumulator is 1024 wide to our 768, their threat space 79,856 dims to
our 60,720 — and they still pay **half** per node. The difference is the
scheme: forward/backward incremental hops on an accumulator stack with a
Finny-style cache, no per-eval replay, no override slots. This is the
engine-side structural item this study surfaces (option 2, §7).

### 3.3 `MovePicker::next_move` — the shell's flagship

| | ARCHON | SF18 |
|---|---|---|
| static instrs | **3,199** (was 2,592 pre-PGO, study 3) | 1,763 |
| calls/node | 2.29 | — |
| top opcodes | `mov` 18.3, `cmp` 13.7, `lea` 11.2, `add` 9.8, `test` 7.2 | `mov` 24.3, `add` 13.5, `lea` 11.8, `sub` 8.1 (study 3, binary unchanged) |
| dyn width | gpr 97.2 | gpr 99.0 |

Same shape as study 3, with PGO having *grown* our copy 23%. Together with
`search::<NonPv>` at **5,168** static instructions (which now carries 5.4%
`vpermi2w` — LLVM vectorized some inlined table math) these two functions are
the icache story: 2.27× SF18's misses/node at only 1.5× the instructions.
SF16's HCE (`Evaluation::value`, 2,623 static instrs, 100% GPR, `pext`/
`popcnt` bitboard idiom) is included in the artifacts for reference.

---

## 4. THE DUAL-NET ADD-ON — full statement

Mechanism (from the sf_18 tag; described, never copied): `simple_eval` =
material-only count diff from stm; `use_smallnet ⇔ |simple_eval| > 962`
(≈4.6 pawns); small net = **L1 128, HalfKAv2_hm only — no threat features**
(`UseThreats == (dims==Big)`); big = L1 1024, HalfKA+FullThreats; if the
small result lands `|nnue| < 277` the node is re-evaluated with the big net.
Both nets keep lazily-updated incremental accumulators; only the selected
one is touched at an eval.

Split, measured exactly (callgrind call counts; the AVX2 rebuild is
node-identical to the official binary — 511,221 = 511,221 at wl50/d10):

| workload | evals/node | small share | small Ir/eval | big Ir/eval | SF18 instr saved |
|---|---:|---:|---:|---:|---:|
| wl50 d10 | 0.440 | **14.4%** | 1,167 | 6,626 | **3.65%** |
| default d10 | 0.438 | **24.0%** | 1,060 | 6,859 | **6.50%** |

Re-stated comparison: forcing SF18 big-always moves our instruction ratio
1.497→1.44 (wl50) / 1.464→1.375 (default), and our cycles ratio by roughly
the same sliver. **The dual net is real, elegant, and a minor part of the
differential.** The far larger "cheap tier" effect in the field is SF16's
classical eval (74–82% of its evals) — which is also why SF16's speed
numbers must never anchor an NNUE-engine target.

**As a lever for us** (§7 option 4): our expensive tier is dearer than SF's
(threats at 768 ≈ 25–30% of instructions), so a threat-free cheap tier fires
on the same 14–24% of nodes and saves proportionally more — an honest ceiling
of **+4–8% NPS** for a real project (a trained threat-free head or small net,
a material predicate, accumulator plumbing, and a fallback band). It is NOT a
patch, and if D49 removes threats entirely it is moot. Parked behind D49's
verdict, with one cheap probe worth doing first: log `|simple_eval|>962`
frequency in our own trees to check our node mix matches SF's (their 24% on
shared middlegame FENs suggests it would).

---

## 5. THE THREAT-COST VERIFICATION — 1.78× refuted, grid established

Method (independent of last night's): `strip_threats.py` on a quant
checkpoint; full AND stripped arms packed from the *same* checkpoint with the
same `netpack` env (so the .cenn pair differs only by threat tensors);
`bench 16 1 300000 <fens> nodes`, arms interleaved per round, node counts
verified equal (15.0M both), instructions:u deterministic (zero spread).
Grid in §0.3; raw rounds in `perf-artifacts-7/threat-strip-grid.txt`.
Sanity: the study .cenn vs the full-repack measures 0.986 i/n — the repack
is pack-equivalent to the shipped net.

Why last night read 1.78×: (a) its 1280 figure came from 3 reps on **2
positions** on a loaded box with the movetime-adjacent single-position
instrument — §1.1 shows exactly that instrument reads ~0.1–0.15 high on
ratio vs matched benches; (b) the stripped net searches a different (garbage-
eval) tree, and per-run NPS without per-node normalization inherits that;
(c) plain load noise — their own log flagged ±2% optimistically while this
session's cycle bands under the same box state are ±8%. The instruction
counter kills the ambiguity: **1.317× at 1280/wl50, 1.389× at
1280/middlegame, 3/3 rounds bit-stable.**

Consequences, restated for the D49 decision (no re-derivation of its Elo
math here, just the corrected speed input): HalfKA-only at the same width
buys **+25% (768/wl50) to +39% (1280/mid) instructions**, cycles somewhat
more on middlegames (up to ~1.5–1.6×), i.e. **~45–70 Elo of speed** at the
house 120/doubling — against an eval value of threats bounded above by
exp-0065's +32.5±17.3 fn (still never isolated). The bet remains live and
positive-EV; its margin is roughly half of what last night's numbers implied.
And per §0.3, even threat-free we remain 1.21× SF18 cycles/node — threats
explain **56% (wl50) / 71% (middlegame) of the instruction excess** over
SF18, not all of it.

---

## 6. WHAT'S STRUCTURALLY LEFT — the hunt

Checked this session, beyond the carried items:

- **`memset`/`memcpy`**: absent from every profile above 0.4% — B4 stays
  MOOT (PGO already deleted the dead zeroing; confirmed on the current
  shipped binary).
- **Startup loader**: 18.2% of short-run Ir (`read_i8s` + `read_i16s` +
  `load_v2_bytes` + `init_magics` at d10) — irrelevant to NPS at depth,
  but it is pure SPRT/screen throughput; the bulk-parse rider from study 6
  stands.
- **`search::<NonPv>` PGO bloat**: 2,689 (v3, study 3) → 5,168 static
  instructions (PGO), `next_move` 2,592 → 3,199. PGO bought −11.9%
  *dynamic* instructions (study 3, verified) at the cost of static size;
  the 2.27× icache differential is now plausibly part-PGO-inflicted. A
  profile-curation pass (train the PGO corpus leaner, or
  `-Z function-sections` cold-splitting when stable) is a small, testable
  item.
- **Evals/node**: ours 0.391 vs SF18's 0.440 — we already evaluate *fewer*
  nodes than SF18; there is no eval-laziness left (study 3 §5.5 confirmed
  standing).
- **SMP scaling: still unmeasured** — the box has not been quiet in any
  session since the question was raised. Open item, release-relevant, needs
  an idle box, do not attempt under load.

---

## 7. OPTIONS RANKED

House convention: removing a fraction f of cycles multiplies NPS by 1/(1−f);
~1 Elo per 1% NPS at 10+0.1 (measured slope has been ~2, these are
conservative).

| # | option | gain | effort | risk | validation class |
|---|---|---|---|---|---|
| 1 | **D49 HalfKA-only retrain** (already staged, `d49_halfka_2048.json`) | **+25–39% instructions measured** (grid §5; ~45–70 Elo speed) minus threats' eval value (≤ +32.5±17.3 fn, unisolated) | GPU-days (trainer ready; 10-line Pallas fix known) | HIGH on eval quality; speed side is now measured, not projected | fn pentagon + ET vs V, the campaign's own gates |
| 2 | ⛔ **SUPERSEDED — see the correction banner at the top.** ~~Retire the `at_full` replay~~ — the rationale below is wrong: 15.7% conflates the PGO-inlined fold with the replay (§3.2's 638 Ir/node is the honest figure), and the fold is irreducible. Addressable surface ≈10.0% of instructions; the +5–8% target is over-promised. Await the implementation agent's measured result. | ~~at_full is 15.7% of deep cycles; SF's equivalent stage costs ~0; honest target **+5–8% NPS**~~ | 1–2 weeks; the hardest engine surgery on the board | HIGH: a missed update is a silently wrong eval | bit-exactness vs the replay path as oracle (`assert_matches_refresh`), 800-FEN hash, node-identity, then SPRT — **plus `CHECK_NNUE_GATE=1`** (the depth-6 bench pin is a PSQT number and cannot see this work) |
| 3 | **Icache/code-size pass** — `next_move` + `search` static-size diet: leaner PGO training corpus, outline cold blocks, de-duplicate tunable materialisation | icache misses 2.27× SF18 on 1.5× instructions; **+2–4%** | days | low-medium | node-identical + icache-counter delta (contention-immune) |
| 4 | **Two-tier eval** (material-gated threat-free tier; §4) | **+4–8% NPS** honest ceiling at SF's measured split | weeks — a trained head, predicate, plumbing, fallback band | medium-high (eval quality on gated nodes) | probe first (predicate freq in our trees), then fn + ET |
| 5 | B6 metadata split + `last_computed` index (carried; `current_full` 8.9% deep) | +1–2% | 1–2 days | medium | bit + node-identical |
| 6 | `skip_dot`/`l2_tail` chain split (carried from study 3 §6.11) | +1–1.5% | half a day | medium (f32 reorder) | int-cp identity on 800 FENs, else SPRT |
| 7 | Threat-PSQ1 i16 rows + quant unification (CENN-rev riders; the shared `vpmovsxbw` tax §3.1 makes this pay on the next format change) | ~+1–2% | rides the next format rev | medium | net-format parity oracle |
| 8 | Loader bulk-read (startup only) | 0 NPS; SPRT/screen throughput | hours | low | 800-FEN hash identity |

Sequence note: 1 is a campaign decision, not a perf patch — this study's
contribution is the corrected speed input. 2–3 are the engine program this
study actually recommends starting, in that order; 5–6 remain the cheap
follow-ons; 4 waits on 1's verdict.

## 8. REJECTED / REFUTED, with the measurement that kills each

| claim / option | the measurement |
|---|---|
| "Threats cost 1.78× at 1280" | 1.317× instructions (deterministic, 3/3 rounds), 1.39× cycles (1.385–1.401), nodes verified equal — §5 |
| "SF18 has no threat channel" | sf_18 `full_threats.h` Dimensions=79,856; `upd_incr<FullThreats,1024>` = 11.3% of SF18's instructions — §0.2 |
| "Strip threats ⇒ SF18-class speed" | stripped arm still 1.21× SF18 c/n, 1.217× i/n, same rounds — §5 |
| "The 1.5× deficit" as a matched-bench fact | 1.31–1.33× on two 50-FEN matched benches; 1.5× is the deep-single-position/movetime end — §1.1 |
| Dual-net as the main SF18 speed explanation | +3.65%/+6.50% of SF18 instructions if removed — §4 |
| SF16's c/n as an NNUE speed target | 74–82% of SF16 evals are classical HCE — §4 |
| `i16_deltas` hoist (perf item 10) | +0.47% instructions; parked-dead on `perf/i16-deltas-hoist` (2026-07-27, re-affirmed, not re-run) |
| B4 dead-zero sweep | libc absent (<0.4%) from every profile of the shipped PGO binary — MOOT stands |
| 2-FEN 1.5M-node ratio arm | startup-contaminated (SF LEB decode); understates by ~0.1 — recorded in artifacts so it isn't re-invented |
| SMP scaling numbers from this box | not rejected — UNMEASURABLE under permanent load; needs the idle box |

## 9. UNCERTAINTIES AND THREATS TO VALIDITY

1. The box was contended throughout (trainer + 4-worker SPSA); every cycles
   band is ±5–10% and quoted as a band; instructions:u (zero-spread),
   callgrind counts, and node totals carry the conclusions.
2. PGO inlines the SIMD kernels into their drivers; block attribution inside
   the eval layer is by symbol and matches the pre-PGO decomposition only
   approximately. The block *totals* (eval vs shell) are robust.
3. The callgrind Ir comparisons run the AVX2 twin paths on both engines —
   schedule counts (calls/node) are path-invariant; Ir ratios between
   engines on that path track but do not equal the native-path instruction
   ratios (native measured separately in §1).
4. The SF18-from-source build is the `sf_18` tag of the reference clone,
   proven node-identical to the official binary at wl50/d10; deeper
   divergence is possible in principle but has no mechanism.
5. Full-vs-stripped arms search different trees (a stripped net evaluates
   badly); per-node normalization removes the first-order effect, but
   second-order tree-shape differences (in-check rate, qsearch share)
   remain in both my figures and last night's.
6. The deep arm is 2 positions; its depth-growth finding is directionally
   solid (three instruments agree) but the exact deep-ratio (1.40×) is a
   2-position number.
7. SF16's official binary is avx2+bmi2; its instruction counts are not
   ISA-matched to our avx512 build (cycles comparisons stand; its i/n is
   if anything overstated, strengthening §4's conclusion).
8. Nothing here was Elo-tested; all Elo figures are the house conversion.

## 10. ARTIFACTS (`perf-artifacts-7/`)

- `counters-matched-fixednode.txt` — all four counter-round groups, absolute
  per-node + per-round ratios.
- `flat-{ours,sf18,sf16}-{deep,d14}.txt` — the six flat profiles.
- `asm/` — opcode/width/mem mixes for the top symbols of all three engines
  (`oursB-*`, `sf18B-*`, `sf16-*`) + two `annotate-*-top25` excerpts.
- `sf-dualnet-split.txt`, `archon-schedule-callgrind.txt` — exact call
  counts, per-eval subtree costs, both mixes; SF16 classical split.
- `threat-strip-grid.txt`, `movetime-reconciliation.txt` — the §5 grid and
  §1.1 reconciliation with raw round values.
- `scripts/` — `pstat7.sh`, `parse7.py`, `opc7.py`, `cgcalls.py`,
  `strip7.sh`, `mt7.sh` (the exact instruments; opc7.py includes the
  load/store classifier new in this study).

Session-local (scratchpad, not committed): perf.data files, callgrind outs,
the v3 twin + SF18-avx2 binaries (`keepbin/`), the four packed net arms.

**Reproduction one-liners.**

```bash
S=<scratch>; awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens > $S/wl50.fens
# identity pins for the study binary+net
./engines/bin/archon-d47v-pgo nnue-eval data/nnue/nets/screens/d47v-a4b2-cal400.cenn \
  < tools/nnue/calib/scale-check-800.fens | sha256sum   # -> dd08810005c92f9e…
taskset -c 4 ./engines/bin/archon-d47v-pgo bench 16 1 13 $S/wl50.fens depth \
  data/nnue/nets/screens/d47v-a4b2-cal400.cenn          # -> 2,118,123
# the matched counter round (per engine, interleave by hand or via scripts/pstat7.sh)
nice -n 19 perf stat -e instructions:u,cycles:u -- taskset -c 4 \
  ./engines/bin/stockfish18 bench 16 1 300000 $S/wl50.fens nodes
# the dual-net split needs the AVX2 rebuild: git archive sf_18 from the SF clone,
# export_net from the official binary, make build ARCH=x86-64-avx2, then
# valgrind --tool=callgrind ./stockfish bench 16 1 10 $S/wl50.fens depth
```
