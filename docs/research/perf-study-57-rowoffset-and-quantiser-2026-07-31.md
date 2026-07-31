# Perf study 57 — pre-scaled fold row offsets, and an `idiv`-free HalfKA quantiser

**Branch** `perf-study-57-i8deltas`, commit `44e5f151`, on top of `c4bcbef3`.
**Box** 7700X / AVX-512 VBMI2, cpu1 (`P57_CPU=1`), shared throughout with the
D49 GPU trainer and perf study 56's `ctl56b` on cpu2.
**Frame** `bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
— 5,001,562 nodes on every binary in this document, and a 1-node variant of the
same command for the startup panels.

**Two arms landed. One arm built and lost, reported with its number. One
finding of my own retracted.**

| | `instructions:u` | `branches:u` | `cycles:u` | `L1-dcache-loads:u` |
|---|---:|---:|---:|---:|
| **arm A** `RowOffset`, 5 M-node frame | **−0.694%** (24/24) | −0.014% (24/24) | −0.099% (13/24, ns) | **+0.283%** (3/24) |
| **arm B** reciprocal quantiser, startup frame | +4.549% (0/39) | **−31.431%** (39/39) | **−22.496%** (39/39) | **−5.856%** (39/39) |
| **both**, 5 M-node frame | **−0.448%** (24/24) | **−2.319%** (24/24) | **−1.363%** (21/24) | +0.052% (7/24, ns) |
| null (byte-identical), 5 M-node frame | −0.000% se 0.000 | −0.000% se 0.000 | +0.298% se 0.324 | −0.086% se 0.030 |

Gates, all green on the landed tree:
`CHECK_NNUE_GATE=1 bash tools/check.sh` → **All checks passed**, `Bench: 587683`,
eval hash `2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4`,
d13 `1739588`; `cargo test --release -- --ignored perft_full_suite` → ok;
node total 5,001,562 on every arm.

---

## 0. Corrections to the commission, up front

**0.1 The profile table.** I re-profiled from source rather than trusting it,
and the *shape* holds — the i8 delta family is the largest block in the engine —
but the symbol split does not match, because it was taken on a different build:

| my re-profile (`sota,arch-d48x`, `target-cpu=native`, fat LTO) | % | commission's table | % |
|---|---:|---|---:|
| `avx512::i8_deltas_into` | **11.66** | `avx512::i8_deltas_into_inner` | 10.36 |
| `search::<NonPv>` | 11.63 | `search::<NonPv>` | 11.19 |
| `avx512::i8_deltas_into_generic` | **10.29** | `avx512::i8_deltas_into_generic` | 8.74 |
| `SotaHead::forward_into` | 8.45 | (`activate_inner::<2>` 3.79 + `l1_dot_inner` 2.65) | 6.44 |
| `MovePicker::init_quiets` | 6.75 | `MovePicker::init_quiets` | 6.35 |
| `Worker::do_move` | 6.53 | — | — |
| `HalfkaContext::finny_refresh_into` | **6.34** | `finny_refresh_into` | **1.56** |
| `ThreatContext::at_full` | 5.35 | `ThreatContext::at_full` | 5.61 |
| `HalfkaContext::current_full` | 4.16 | `HalfkaContext::current_full` | 3.72 |
| `sota_loader::load_v2_bytes` | 2.50 | `sota_loader::load_v2_bytes` | 2.70 |
| `sota_loader::quantise_halfka` | 2.12 | `sota_loader::quantise_halfka` | 2.42 |
| — | — | `avx512::i8_deltas_dual_inner` | 5.00 |

`i8_deltas_dual_inner` **does not appear at all** in my profile, and
`finny_refresh_into` is 4× larger — the dual kernel is inlined into its one
caller in this build. Same for `activate_inner`/`l1_dot_inner` inside
`forward_into`. Nothing is wrong with either reading; they are different
inlining decisions, and the lesson is that a symbol share is a property of a
build, not of the engine.

The i8 delta family is **21.95%** here against the commission's 24.10%.

**0.2 "This block has not been touched at all" is false, and materially so.**
Studies **21, 25, 32, 36, 37, 43, 44, 46, 47, 48 and 51** all worked these
kernels. Eight software-prefetch arms have died here; the tile width is a
closed sweep; i16 rows, nibble packing, zero-row skipping, add/sub cancellation,
non-temporal stores, huge pages and PSQ1 interleaving each have a number
against them. I checked the catalogue before building, which is the only
reason arm A exists — it is the one candidate a prior study *named and did not
build*.

**0.3 `aligned.rs`'s stale comment is real** (line 19, an `i16_deltas_into`
30.3% profile from the pre-study-32 era) — the commission's one incidental
claim that survived.

---

## 1. What the `_inner` / `_generic` split actually is — measured, not assumed

The commission's primary lead was that a generic fallback burning 8.74%
alongside a specialised inner "suggests the specialised path is not covering
the cases it should". I measured the call-site split with the `foldcount`
instrumentation that already ships (`--features foldcount,rowprobe`), exact
integers, study frame:

```
hka_into.calls=4893314   hka_into.rows=11150925    (2.279 rows/call)
hka_dual.calls=823842    hka_dual.rows=4712390     (5.720 rows/call)
hka_inplace.calls=128    hka_inplace.rows=645
thr_into.calls=4270334   thr_into.rows=19616896    (4.594 rows/call)
thr_refresh.calls=98024  thr_refresh.rows=407362
hka_shape.1a1s=3529017  1a2s=1316030  2a1s=48267   (sums to hka_into.calls EXACTLY)
```

**The split is correct, and it is not a coverage failure.**

* `hka_shape` sums to `hka_into.calls` to the unit: **100% of HalfKA
  into-folds land on a shaped instantiation.** `(2,2)` is still n=0 (study 36's
  finding reproduces; it is castling, which routes to a Finny refresh).
* The generic path is **the FullThreats channel**, at 4.594 rows/call against a
  shape table that only covers `(≤2, ≤2)`. Study 25 §3 measured the threat mean
  at 1.98 adds / 2.66 subs with ~100 distinct shapes, so most threat folds miss
  the table by construction.
* Rows per node: HalfKA into 2.230, threat into 3.922, Finny dual 0.942, threat
  refresh 0.081 — **7.175 total**, reproducing study 47's 7.1754 exactly.
  **The threat channel is 54.7% of all folded rows.**

So the answer to the lead is: *the generic path is genuinely necessary for its
cases*, and the commission's instruction in that event was to make **it**
faster. That is arm A.

**And the obvious alternative was already built and lost.** Study 25 §3
implemented exactly the shape dispatch this lead points at, for the threat
channel, at `{0,1,2}²` (52% coverage): **+0.14% instructions, +0.24% for the
packed-key variant, hot code 3336 → 11674 bytes**, cycles inside the null. I
did not rebuild it. §6 re-prices it, because the price has moved a lot and it
is now the best-named follow-up.

---

## 2. RETRACTION — the "`instructions:u` bimodality" I reported to myself

Mid-study I found `instructions:u` taking one of exactly two values 206.4 M
apart (0.694%) for **byte-identical binaries**, at a ~33% rate, with the node
total unchanged. That would have been a first-order finding: it is the counter
studies 25–56 adjudicate on, on the stated grounds that its null is 0.000.

**It was not real. I caused it.** An earlier `nohup … &` launch of the panel
outlived the tool call that started it and kept appending to the same output
file while the replacement panel ran — two jobs, one file, and both benches on
cpu1. The parser then assigned one job's `perf stat` records to the other job's
round markers, scrambling labels ~50/50. The two "modes" were simply the base
values and the arm values.

What settled it was the cheapest possible instrument, `tools/perf57/pair57.sh`
— two labels, two events, one line per run, alternating:

```
rowoff   29536323884    3101156384    nodes=5001562
base     29742750867    3101606179    nodes=5001562
base     29742752166    3101606891    nodes=5001562
rowoff   29536323418    3101156242    nodes=5001562
```

Perfectly separated by label. `instructions:u` is exact on this box, run to run,
to about 1500 instructions in 29.7 G (5 × 10⁻⁸).

Three things worth keeping:

1. **The interleaving hazard is in the inherited harness shape.**
   `tools/perf55/exact55.sh` (and 48/45 before it) appends the round marker,
   `perf stat`'s CSV *and* the engine's own stderr to one accumulating file.
   That is correct for one job and silently wrong for two, and nothing in it
   detects the second job. `tools/perf57/panel57.sh` fixes both halves:
   `perf stat -o` writes to a private temp file per run so only parsed numbers
   reach the panel, and an `flock` on `$OUT.lock` makes a second panel refuse.
   The node total is now carried on the marker and `med57.py` prints the set,
   so a panel that is not node-identical announces itself.
2. **A beautiful mechanism was available for the wrong result and I nearly took
   it.** I had a plausible story (PMU counter allocation, SMT sibling effects)
   and a filter half-written before checking the raw file. Rule 6.
3. **The tell was in the data the whole time**: in the "low mode",
   `branches:u` moved −0.014% while instructions moved −0.694%. No real code
   has 475 instructions per branch. I wrote that down and kept going anyway.

---

## 3. Arm A — `RowOffset`: the fold kernels take pre-scaled row offsets

### 3.1 The mechanism, off the shipped disassembly

`i8_deltas_into_generic`'s row loop, `bin57/base`, `objdump`:

```
76370: mov    (%r9),%r15          ; load the feature index
76373: add    $0x8,%r9            ; cursor
76377: lea    (%r15,%r15,4),%r15  ; ×5    ⎫  feature * ACC
7637b: shl    $0x8,%r15           ; ×256  ⎭  (ACC = 1280 = 5 · 256)
7637f: inc    %r8                 ; trip counter
       5 × vpmovsxbw  /  5 × vpaddw
763c8: jne    76370
```

Six scalar instructions per row **per tile**. The tile count is the point:
`REGS = 5` (study 44's landed `REGS_CAP`) and `ACC_BLOCKS = 40`, so
`TILE = 160` lanes and **`TILES = 8`**. `feature * ACC` is therefore recomputed
**eight times per folded row**.

Study 37 §5 named this candidate and priced it: *"pre-scaled row offsets (store
`feature * ACC` at resolve time, kill the `lea`+`shl`) — arithmetically −8 i/n
(−0.135%) … left as the best-priced remaining candidate."* **That price was
computed at `TILES = 2`.** Study 44 then landed `REGS_CAP = 5` and made the walk
4× wider, and nobody re-derived the number. Study 36 §6 had independently
declined the neighbouring shape-dispatch arm on a "4 instructions per row per
tile at TILES = 2" estimate — the same stale tile count.

Static prediction at `TILES = 8`: 2 instructions × 8 tiles × 3.922 threat
rows/node = −62.8, plus 2 × 8 × 0.942 Finny rows = −15.1, minus ~2 per row at
the producer ≈ **−63 i/n**.

### 3.2 The change

`RowOffset(usize)` in `sota_features.rs`, `RowOffset::of(f) = f * ACC`. Every
producer emits it; every kernel indexes straight off `.0`.

Producers: `resolve_both` / `resolve_into` (threats), `apply_dirty_directed`
(HalfKA into), the Finny warm and cold refreshes. Kernels: `i8_deltas`,
`i8_deltas_into`, `i8_deltas_dual`, `i8_into_shaped` and the generic fallback,
in all three backends.

**Why a newtype and not a bare `usize`.** Nine sites have to agree on the unit,
the two channels share one kernel (`apply_threat_deltas_into` and
`apply_halfka_deltas_into` both dispatch to `i8_deltas_into`), and a silent
disagreement is an evaluation that is arbitrarily wrong and still benches. The
compiler found every site; the eval-hash gate then confirmed it. This is the
idiomatic-Rust answer to exactly the failure mode the NNUE gate exists for.

PSQ1 rows are `BUCKETS` wide, not `ACC`, so `psqt_add`/`psqt_sub`/`psqt_fold`
keep the plain feature index. Three sites therefore carry two parallel short
lists; the threat resolver needed none, because it accumulates PSQ1 inline from
the local index it already has.

A bonus deletion: the avx2 backend's `i8_deltas_into_inner` carried a
`[MaybeUninit<usize>; 96]` pair plus a chunking loop (perf study 12 §6-F) whose
only purpose was hoisting this multiply. It is gone, and with it the last
`MaybeUninit` in that backend.

### 3.3 Result — `docs/research/perf-artifacts-57/panel-arm.txt`

24 rounds, blocked Latin square, `base` / `null` (byte-identical copy) /
`rowoff`, all node-identical at 5,001,562, mean position 1.00 for all three.

```
label   event                     median %  trimmed %     se   neg/n      p
null    instructions:u              -0.000     -0.000  0.000   17/24  0.0639
null    branches:u                  -0.000     -0.000  0.000   18/24  0.0227
null    cycles:u                    -0.399     -0.294  0.219   17/24  0.0639
null    L1-dcache-loads:u            0.003     -0.006  0.030   11/24  0.8388
null    ls_dispatch.ld_dispatch:u    0.043      0.050  0.052    9/24  0.3075

rowoff  instructions:u              -0.694     -0.694  0.000   24/24  0.0000
rowoff  branches:u                  -0.014     -0.014  0.000   24/24  0.0000
rowoff  cycles:u                    -0.099      0.009  0.279   13/24  0.8388
rowoff  L1-dcache-loads:u            0.283      0.265  0.047    3/24  0.0003
rowoff  ls_dispatch.ld_dispatch:u    0.020      0.001  0.076   10/24  0.5413
```

**−0.694% instructions = −41.27/node** (5946.69 → 5905.42), 24 of 24 rounds,
zero spread, against a null of exactly 0.000 with zero spread. **−0.014%
branches**, also 24/24 at zero spread.

The row loop is now four scalar instructions instead of six; the disassembly
confirms `lea` and `shl` are gone and `vpmovsxbw` count is unchanged at 450.

### 3.4 The regression, stated plainly

**`L1-dcache-loads:u` is +0.283% (3/24 negative, se 0.047, p = 0.0003)** —
about **+8.3 loads/node**, against a null of +0.003% and a documented floor of
0.04–0.25% on this counter. It is real and it is a cost of the change: three
sites now carry a second short list so PSQ1 keeps its feature index, and LLVM
materialises both.

I did not chase it, and here is the arithmetic rather than a shrug. Study 48
§1.1 measured this engine's load classes: weight rows 1.40 stall cycles per
load, scalar search ~0.67, **accumulator/stack ~0.29**. These are stack loads,
the cheapest class, so +8.3 of them is ≈ **+2.4 cycles/node** against −41.3
instructions/node. The one alternative I costed — teaching `psqt_fold` to take
`RowOffset` and recover the feature with a divide-by-constant — trades the 8.3
loads for **+14 instructions/node**, i.e. it pays the counter with a 0.000 null
to relieve a counter with a 0.04% floor. Left as is.

`cycles:u` is **not resolvable** for this arm and I do not claim it: −0.099% at
13/24 against a null of −0.399% at 17/24. Both are inside the box's cycles
floor. Arm A is an exact-counter win.

### 3.5 What was predicted and what was measured

Predicted −63 i/n, measured −41.3. The 21-instruction gap is the shaped path:
threat folds with 1–2 rows in a `(≤2, ≤2)` split *do* reach `i8_into_shaped`,
which already hoisted the multiply once per call, so those rows were never
paying 8×. The prediction assumed every threat row went generic. The
**direction, mechanism and order of magnitude hold; the size was over-priced
1.5×** and I record that rather than round it away.

---

## 4. Arm B — `quantise_halfka` without `idiv`

### 4.1 Startup is bigger than the symbol shares say, and here is the exact number

Study 36 §4 read startup off symbol shares (5.2%) and listed as an
uncertainty that it *"did not run the second budget that would have made the
number exact."* That budget costs one command. A 1-node bench against the
5 M-node bench, same binary, same net:

| | 1-node (startup) | 5 M-node frame | startup share |
|---|---:|---:|---:|
| `instructions:u` | 1.611 G | 29.743 G | **5.42%** |
| `cycles:u` | 1.196 G | 18.158 G | **6.58%** |

Startup is **6.58% of the cycles of the pinned bench**, and every one of them
is paid again by every process a match or a screen spawns.

### 4.2 The change

`quantise_halfka` performs `HALFKA_DIMS · ACC = 28,835,840` integer divisions,
one per weight — `round_half_away(w, scale)` is `(2·|w| + s) / (2·s)`, and `s`
is a runtime value so LLVM must emit a real `div` (four of them survive in the
shipped `bin57/base` symbol, plus a `panic_const_div_by_zero` edge). `div` is
the one common integer op Zen 4 does not pipeline.

Replaced by the standard round-up reciprocal: `M = ⌊2³²/D⌋ + 1`, `q = (N·M) >> 32`.

**This is exact, not approximate, and both domain bounds are structural.**
`D = 2·scale ∈ [2, 510]` because `halfka_column_scales` clamps a `u8` to
`[1, 255]`; `N = 2·|w| + scale ≤ 2·32768 + 255 = 65,791` because `w` is an
`i16`. The requirement is `N < 2³²/e` with `e ≤ D`, and `2³²/510 = 8,421,504`
— a **128× margin**. Two new tests sweep *every* `(scale, N)` pair and *every*
`(w, scale)` pair against the real division rather than trusting the algebra;
both run in under half a second.

### 4.3 Result — `docs/research/perf-artifacts-57/startup2.txt`

39 complete rounds on the 1-node frame, `rowoff` (unmodified quantiser) as
reference:

```
label  event                     median %  trimmed %     se   neg/n      p
all2   cycles:u                   -22.496    -22.504  0.191   39/39  0.0000
all2   branches:u                 -31.431    -31.431  0.000   39/39  0.0000
all2   ls_dispatch.ld_dispatch:u  -16.203    -16.157  0.064   39/39  0.0000
all2   L1-dcache-loads:u           -5.856     -5.830  0.048   39/39  0.0000
all2   instructions:u              +4.549     +4.549  0.000    0/39  0.0000
```

**−22.5% of startup cycles, 39 of 39 rounds**, on a counter that on this box
normally cannot resolve anything — the effect is 118× its own standard error.
Instructions go **up** 4.55%: that is the whole point of the trade, and it is
the one place in this study where `instructions:u` would have given the wrong
verdict. `branches:u` −31.4% is the `div`-by-zero edge and the sign branches
going away.

In absolute terms: **−269 M cycles per engine process**, ≈ 62 ms at this box's
clock. On the pinned bench that is −1.48% of the whole frame.

### 4.4 The negative arm — the row-major rewrite LOSES, and by a lot

My first implementation also removed the `i % ACC` column recovery by walking
the table row-outer (`raw.chunks_exact(ACC*2).flat_map(|row| …zip(scale)…)`).
That reads like a strict improvement: the modulo disappears and `scale` becomes
a sequential 1280-element stream. Measured against the *same* reciprocal, 40
paired rounds (`startup.txt`, then reconfirmed at 39 rounds alongside the
survivor in `startup2.txt`):

```
all    instructions:u             +10.668    +10.668  0.000    0/40  0.0000
all    branches:u                 +19.018    +19.018  0.000    0/40  0.0000
all    L1-dcache-loads:u          +36.661    +36.535  0.222    0/40  0.0000
all    ls_dispatch.ld_dispatch:u  +22.572    +22.567  0.207    0/40  0.0000
all    cycles:u                    +5.294     +5.372  0.211    0/40  0.0000
```

**A loss on every counter, 0/40 on all five, including cycles.**

The mechanism is the collector, not the arithmetic. `Aligned::from_iter_len`
ends in `Vec::extend`; a `FlatMap` is neither `TrustedLen` nor exact-size, so
`extend` falls off its bulk specialisation onto the per-element
`extend_desugared` path — a capacity check per weight, 28.8 M of them. **A
modulo per element is cheaper than losing the `extend` specialisation.** That
is now a ⚠ banner on the function, because nothing in the source suggests it.

This is also the reason arm B is reported as two separate builds. Had I shipped
the combined version I would have landed a 5.3% startup *regression* while
believing I had removed a division.

---

## 5. The combined tree, and what it is worth

`docs/research/perf-artifacts-57/panel-final.txt`, 24 rounds, node-identical:

```
label  event                     median %  trimmed %     se   neg/n      p
null   instructions:u              -0.000     -0.000  0.000   12/24  1.0000
null   branches:u                  -0.000     -0.000  0.000   15/24  0.3075
null   cycles:u                    +0.298     +0.241  0.324   11/24  0.8388

all2   instructions:u              -0.448     -0.448  0.000   24/24  0.0000
all2   branches:u                  -2.319     -2.319  0.000   24/24  0.0000
all2   ls_dispatch.ld_dispatch:u   -0.556     -0.578  0.037   24/24  0.0000
all2   cycles:u                    -1.363     -1.187  0.388   21/24  0.0003
all2   L1-dcache-loads:u           +0.052     +0.049  0.023    7/24  0.0639
```

The parts add up, which is the cross-check: arm A's −0.694% of search
instructions plus arm B's +4.549% of a 5.42% startup share (+0.246%) predicts
−0.448% — measured −0.448%. Branches: −31.431% of a 7.34% startup share
predicts −2.31%, measured −2.319%.

**On cycles the honest attribution is that essentially all of the −1.363% is
arm B.** −22.496% of a 6.58% share predicts −1.48%; measured −1.363%. Arm A's
own cycles contribution is not resolvable and I am not claiming it. This is the
one panel in the study where `cycles:u` carries a verdict, and only because
21/24 with p = 0.0003 against a null at 11/24 is a sign test, not a median.

**What it is worth in Elo.** Arm B is a fixed per-process cost, so its value
depends entirely on the process lifetime: ~62 ms of a game (negligible at any
real TC), but a full −1.48% for every fixed-node bench, screen and datagen
invocation, of which `tools/screen.sh` runs thousands. Arm A is −41 i/n whose
cycles value this box cannot measure. **Neither arm is claimed as Elo, and
neither has been game-tested.** At ARCHON's own ~122.9 Elo per speed doubling a
1.4% frame saving is ~2.5 Elo — six figures of games to see directly, which is
why the exact counters are the arbiter here.

---

## 6. Priced and not taken, each with its number

| candidate | why not, with the number |
|---|---|
| **Shape-specialise the threat fold** — **the best-named follow-up, and its price has moved twice** | Study 25 §3 built it at `{0,1,2}²` and measured **+0.14% instructions**, hot code 3336 → 11674 B, at `TILES = 2`. Study 36 §6 re-declined it on an estimate of "32 i/n" — also at `TILES = 2`. At today's `TILES = 8` **and after arm A**, the remaining per-row control is 4 instructions × 8 tiles × 3.922 threat rows = **125 i/n = 2.1% of engine instructions**, while the dispatch cost is per-CALL and unchanged at ~0.56%. Expected net ≈ **−1.5% instructions**. It is one build, and the falsifier is study 25's own: if instructions rise, the coverage argument beat the control argument again. I ran out of budget, not out of case. |
| **`psqt_fold` on `RowOffset`** (would delete arm A's +0.283% load regression) | +14 instructions/node for −8.3 loads/node. Pays a 0.000-null counter to relieve a 0.04%-floor counter, at ~0.29 stall cycles per load (study 48 §1.1). Priced, declined. |
| **The FNV-1a-64 payload checksum**, the largest single startup item left | 135,578,656 bytes on a serial `xor`→`imul` chain = 4 cycles/byte ≈ **542 M cycles**, i.e. roughly **45% of all startup cycles** and bigger than arm B. It is not reducible without changing the value: FNV's recurrence is `h = (h ⊕ b)·P`, the XOR only touches the low byte, and the low-byte sub-chain is itself 4 cycles/byte (or an L1 lookup, no better). Interleaved sub-streams would change the checksum, i.e. the container format. **Reported, not built** — the only routes are a format change or skipping the check, and both need a ruling. |
| **A pre-quantised i8 net** (the commission's secondary suggestion) | Would delete `halfka_column_scales` (57.7 MB read) *and* `quantise_halfka` *and* 21% of the FNV payload. But it changes the CENN container **and the shipped canonical net file**, and `c30611fd` rules the net out of agent scope. `sota_loader::serialize` is already idempotent (`load(serialize(load(x))) == load(x)`), so it is buildable and bit-checkable — it needs a human decision, not more measurement. Naming it, not doing it quietly. |
| **The 77.7 MB threat-table copy** (`read_i8s_aligned`) | ~155 MB of traffic ≈ 13 ms, against arm B's 62 ms. Avoiding it means keeping the file bytes alive and taking an aligned view, which the 64-byte-alignment contract does not permit in general. |
| **Software prefetch of weight rows** | Dead eight times (studies 4, 21, 29 ×3, 43 ×4, 44, 51). Not re-run. |

---

## 7. Instrument notes for the next study

* **`instructions:u` and `branches:u` are exact on this box** — 0.000% median at
  0.000 se on byte-identical binaries, in three separate null panels here. The
  retraction in §2 was my harness, not the counter.
* **`cycles:u` is not.** Nulls of +0.298%, −0.399% and +0.711% at 12–24 rounds,
  se 0.22–0.52. It resolved exactly once in this study, at an effect of −1.363%
  with 21/24 sign agreement. Below ~1% it should not carry a verdict at these
  round counts.
* **`L1-dcache-loads:u` floor** measured here at 0.003–0.086% median, se
  0.023–0.042. `ls_dispatch.ld_dispatch:u` at 0.043–0.122%, se 0.052–0.099.
  Both usable at effects above ~0.2%; note they disagreed on arm A (+0.283% vs
  +0.020%), so do not treat them as the same event.
* **Never let two panels share an output path.** `panel57.sh` now takes an
  `flock` and writes only parsed numbers, and `med57.py` prints the set of node
  totals it saw. `tools/perf55/exact55.sh`'s shape has neither guard.
* **Kill background jobs by PID and check they are dead.** `strays57.sh` lists
  this study's processes with the `shell-snapshots` self-exclusion.

---

## 8. Reproduce

```bash
bash tools/perf57/build57.sh base                    # arm binaries into bin57/
cp bin57/base bin57/null                             # the inert control
bash tools/perf57/build57.sh rowoff                  # (from the arm-A commit)
P57_CPU=1 P57_PANEL=docs/research/perf-artifacts-57/panel-arm.txt \
  bash tools/perf57/panel57.sh 24 base null rowoff
python3 tools/perf57/med57.py docs/research/perf-artifacts-57/panel-arm.txt base null rowoff

P57_BUDGET=1 P57_PANEL=docs/research/perf-artifacts-57/startup2.txt \
  bash tools/perf57/panel57.sh 40 rowoff all all2    # the startup frame
P57_NODES=1 python3 tools/perf57/med57.py docs/research/perf-artifacts-57/startup2.txt rowoff all all2

bash tools/perf57/pair57.sh 6 base rowoff            # the minimal settling instrument
CHECK_NNUE_GATE=1 bash tools/check.sh                # the gate
bash tools/perf57/perft57.sh                         # the D24 release tier
```

The `foldcount` census in §1 is
`bash tools/perf57/build57.sh foldcount foldcount,rowprobe` then
`ARCHON_FOLDCOUNT=… ./bin57/foldcount bench 16 1 100000 …`.
