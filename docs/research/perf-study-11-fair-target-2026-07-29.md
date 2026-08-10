# ARCHON perf study #11 — the fair target, and the instrument that lied

> **2026-08-10 (license audit R2-1):** the SF-derived artifacts this study cites
> (`sfcount11-nnue_accumulator.cpp.txt` — a verbatim GPL source copy — and the
> `*sfeca*` disassembly listings) were removed from the tracked tree; regenerate
> per `perf-artifacts-11/README-regenerate.txt` if ever needed.


Date: 2026-07-29. Eleventh commissioned study (8:
`perf-study-8-nps-parity-2026-07-28.md`, 9: `perf-study-9-parity-items-2026-07-28.md`,
10: `perf-study-10-parity-floor-2026-07-28.md`). Raw artifacts and instruments:
`perf-artifacts-11/`. Baseline: `main` at `51fe78e` (contains study 10's merge
`a0b2bc9`). Comparison target: **`engines/bin/stockfish-eca43a97`**.

**Nothing landed in `src/`.** Two implementations were written, measured, and
reverted as nulls (§5). The study's product is a corrected target, a corrected
accounting, and **the retraction of the measurement instrument that three
previous studies used for every cross-engine eval comparison** (§4).

---

## 0. HEADLINE

### 0.1 The honest ratio

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, three arms interleaved,
`taskset -c 4`, `nice -n 19`, 6 rounds, ARCHON arm PGO. Raw:
`counters-headline.txt`. Box: load 7.5–9.6 throughout (a D48Y trainer plus TM
matches), 3–5 GB free, swap full.

| per node | **ARCHON** (`archon-s10-final`) | **SF `eca43a97`** | SF18 |
|---|---:|---:|---:|
| **instructions** | **6367.2** | **4452.4** | 4329.6 |
| **ratio vs `eca43a97`** | **1.4301×** | — | 0.9724× |
| branches | 734.2 | 362.7 | 394.7 |
| branch-misses | 23.2 | 20.6 | 26.7 |
| nodes | 15,001,568 | 15,008,494 | 15,010,238 |
| `instructions:u` spread | 0.0018% | 0.0000% | 0.0000% |

The instrument is validated by its own control: **SF18 reproduces 4,329.6
exactly** — study 10's figure, and the seventh independent session to land on
it. ARCHON reproduces study 10's 6,367.1 to 0.002%.

**The ratio against a fair, single-net target is 1.4301×, not 1.4706×.**

### 0.2 The target is verified single-net, and verified to be our own workload

* `stockfish-eca43a97 uci` exposes **one** `EvalFile` option and **zero**
  occurrences of "small". Study 10 flagged this as unverified; it is now
  verified. The small-net term does not apply and appears nowhere in this
  study's accounting.
* Its net is `(83248, 1024, 32, 32, 1)`. **83248 = 22528 + 60720** — the
  HalfKAv2_hm + FullThreats pair at exactly ARCHON's dimensions.
* **SF18 is a different workload.** Its big net is `(102384, 1024, 15, 32, 1)`
  and its `FullThreats::Dimensions` is **79856**
  (`sf_18:src/nnue/features/full_threats.h:45`) against `eca43a97`'s **60720**
  (`:41`) — a **31% larger threat feature set**, plus an L2 of 15 against 32.
  Every study 8/9/10 threat comparison was made against that.

### 0.3 How much of study 10's "unreachable" argument survives

Study 10 priced SF18's adaptive small-net blend at **325–740 i/n, "20% to 44%,
centre ≈31%" of the 1,667 i/n it was asked to remove**, and called it "a feature
we lack, not inefficiency we carry".

Measured against the engine that actually has no small net:

| | study 10 (vs SF18) | **this study (vs `eca43a97`)** |
|---|---:|---:|
| target i/n | 4329.6 | **4452.4** |
| 1.10× bar | 4762.6 | **4897.6** |
| required reduction | −1667.3 | **−1469.6** |

**Removing the small net moved the bar by 197.7 i/n — 11.9% of study 10's
required reduction, not 20–44%.** The excuse was real in kind and over-priced
by 1.6× to 3.7× in size.

[Caveat, stated up front: `eca43a97` differs from SF18 in three ways at once —
no small net, a 24% *smaller* threat feature set, and a 2.1× *wider* L2. The
198 i/n is the net effect of all three, not an isolation of the small-net term.
It is nonetheless the right engineering target, because it is the engine whose
feature set we provably share.]

**So most of study 10's argument survives, and the part that survives is now
harder**: 1,469.6 i/n must come out with no structural excuse available at all,
and §4 shows the largest block study 10 pointed at has nothing in it.

### 0.4 The finding that matters most

**The `perf annotate` per-opcode instrument that studies 8, 9 and 10 used for
every cross-engine eval comparison is wrong by up to 1.44× on this box, and
this study's own headline hypothesis was built on it and then destroyed by a
deterministic counter.** Details in §4. This is the reason nothing else in this
document is quoted from a sampled opcode share.

---

## 1. METHOD

Studies 8–10's rules were followed. Three things were added or changed:

**NULL FIRST.** Study 10 established that the PGO pipeline contributes exactly
zero build-to-build variance (two byte-identical binaries from identical
source). That null is inherited, not re-run. The null this study *did* run is
the one that mattered: **before believing a cross-engine opcode ratio, count the
same quantity deterministically on at least one side.** It failed (§4).

**`instructions:u` only.** The box carried a D48Y training run and TM matches
all session. Per-round paired NPS on byte-identical binaries searching
byte-identical trees ranged **0.6253–0.6801** (ARCHON vs `eca43a97`) while
`instructions:u` reproduced at 0.0018%. **No cycles or NPS number in this
document is offered as a claim.**

**Probe pairs for per-change pricing.** `scripts/probe11.sh` builds the working
tree native non-PGO and measures a fixed 5,001,562-node tree, printing the wl50
eval hash so a tree or eval change cannot hide inside a speed claim. Identical
source + identical flags ⇒ identical binary ⇒ the difference between two labels
is exact, not statistical.

### 1.1 Gate status

Nothing landed, so no pin moved. A from-source `sota,arch-d48x` release build of
the final tree reproduces both NNUE-gate pins exactly:

```
eval hash  5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38
d13 nodes  1922947
```

---

## 2. THE DETERMINISTIC WORKLOAD COMPARISON

Because ARCHON and `eca43a97` provably compute the **same feature set at the
same dimensions**, the number of accumulator rows folded per node is directly
comparable — and it is a *workload* quantity, so it is immune to codegen, PGO,
sampling and box load.

Both engines were instrumented with atomic counters and run on the identical
bench (`bench 16 1 30000 wl50.fens nodes`). ARCHON's patch:
`counters11.patch` (reverted). SF's: a scratch copy at `/home/nick/sfcount11`,
non-PGO — row counts do not depend on codegen. Both patches are archived in
`perf-artifacts-11/`.

| per node | **ARCHON** | **SF `eca43a97`** | ratio |
|---|---:|---:|---:|
| nodes in run | 1,500,780 | 1,501,025 | — |
| threat **delta** rows | 4.0481 | 3.8005 | 1.07× |
| threat **refresh** rows | 0.1609 | 0.9626 | **0.17×** |
| **total threat rows** | **4.2089** | **4.7631** | **0.88×** |
| threat perspective-folds | 1.2005 | 0.9982 | 1.20× |
| — of which fold ZERO rows | 0.3355 | — | — |
| threat refreshes | 0.0352 | 0.1673 | **0.21×** |
| HalfKA/PSQ delta rows | (not measured) | 2.2585 | — |
| events generated | 2.2152 | — | — |
| plies whose events were generated | 0.6206 | — | — |
| pushes (snapshots taken) | 1.0087 | — | — |

**We fold FEWER threat rows than Stockfish — 0.88× — on an identical feature
set.** And our lazy king-cross refresh scheme is dramatically better: SF
refreshes 4.8× as often and pays 6.0× the refresh rows.

Converting to instructions with each kernel's *exact* shape — ours is a
5-pass × 8-register tile over 1280 i16 lanes ⇒ **40 `vpmovsxbw` per row**
(`sota_simd.rs:1286-1309`); SF's is 1024 lanes ⇒ **32 per row**:

| | rows/node | widen i/n | ratio |
|---|---:|---:|---:|
| ARCHON | 4.2089 | **168.4** | — |
| SF `eca43a97` | 4.7631 | **152.4** | **1.10×** |

**The threat fold is at parity**, and the 1.10× residual is entirely the
1.25× accumulator width (1280 vs 1024) partially offset by our 0.88× row count.

**Study 10's conclusion here was right; its evidence was not** (§4). Three
consecutive studies have now failed to find money in this channel, and this is
the first time the verdict rests on a count rather than a sample.

---

## 3. BLOCK ACCOUNTING vs `eca43a97`

Instruction-attributed flat profiles (`perf record -e instructions:u -c 2000003`).
ARCHON's is **non-PGO** (6946.5 i/n), SF's is **PGO** (4452.4 i/n).
Raw: `flat-s11base-instructions.txt`, `flat-sfeca-instructions.txt`.

**Frame warning, inherited from study 10 §6.1 and re-confirmed:** non-PGO leaves
small callees outlined that PGO inlines into `search`. ARCHON's eval block reads
**45.4% non-PGO** against study 10's **52.7% PGO**. Every ARCHON row below is
therefore a **lower** bound in the PGO frame; quote the eval block as
**3,155–3,355 i/n**, and no single-symbol split to better than ±6%.

| block | ARCHON i/n | SF i/n | ratio | excess |
|---|---:|---:|---:|---:|
| **eval layer** | **3155** | **1975** | **1.60×** | **+1180** |
| movepick | 781 | 675 | 1.16× | +106 |
| make / unmake | 505 | 262 | 1.93× | +243 |
| search shell | 1080 | 1065 | 1.01× | +15 |
| **total measured** | **5521** | **3977** | — | **+1544** |
| unattributed (below 0.05% cut) | 1426 | 475 | — | +371 |
| **engine** | **6947** | **4452** | **1.56×** | — |

Constituents (share of engine, then i/n):

* ARCHON eval — `at_full` 16.44% (1142), `forward_into` 8.87% (616),
  `current_full` 7.16% (497), `finny_refresh_into` 5.42% (376),
  `SotaContext::evaluate` 1.92% (133), `record_targets` 1.59% (110),
  `record_square_touch` 1.52% (106), `SotaContext::push` 1.46% (101),
  `refresh_threat` 1.05% (73).
* SF eval — `Network::evaluate` 29.99% (1335),
  `update_accumulator_refresh_cache` 6.34% (282),
  `update_piece_threats<true>` 6.15% (274), `<false>` 1.38% (61),
  `apply_combined` 0.50% (22).
* The make/unmake row is like-for-like: **both** engines' threat *event
  generation* is counted in the eval block (SF's `update_piece_threats`, ours
  `record_*`), so neither side's `do_move`/`make_move_with` is carrying it.

**The eval layer is 62% of the entire 1,915 i/n gap.** Inside it, the fold is at
parity (§2), so the excess is the head, the HalfKA path, and the threat scalar
plumbing — in that order of size and in inverse order of how well any study has
characterised them.

**`search` is at parity** (1.01×), which is worth recording because study 10's
table had it at 1.16× against SF18. **`movepick` is not** at parity (1.16×),
against study 8's "already at parity" finding vs SF18.

---

## 4. THE INSTRUMENT RETRACTION

### 4.1 What happened

Mid-study, the same `perf annotate` opcode method studies 8–10 used produced a
beautiful result: `vpmovsxbw` — the i8→i16 widen only the FullThreats fold
performs — measured **226.3 i/n whole-engine in ARCHON against 142.3 in
`eca43a97`**, i.e. **1.59×**, on an identical feature set. That would have
refuted study 10's parity finding and re-opened "make the threat channel
incremental" as the largest available item.

It was wrong. The deterministic counters (§2) give **168.4 vs 152.4 = 1.10×**.

| | sampled | deterministic | error |
|---|---:|---:|---:|
| ARCHON `vpmovsxbw` | 226.3 | 168.4 | **+34.4%** |
| SF `vpmovsxbw` | 142.3 | 152.4 | −6.6% |
| **ratio** | **1.59×** | **1.10×** | **1.44×** |

### 4.2 Why

Non-precise sampling piles a loop body's samples onto particular instructions.
The cleanest demonstration is ARCHON's `l1_dot`, whose 8×-unrolled body is
literally four instructions repeated:

```
4.63 : movzwl 0x2(%rdx),%r10d
0.00 : vpbroadcastd 0x120(%rsp,%r10,4),%zmm1
0.03 : shl    $0x6,%r10d
0.00 : vpdpbusd (%r8,%r10,1),%zmm1,%zmm0
6.82 : movzwl 0x4(%rdx),%r10d
0.00 : vpbroadcastd 0x120(%rsp,%r10,4),%zmm1
...
```

Every `movzwl` carries 4.1–6.8%; every `vpdpbusd` and `vpbroadcastd` next to it
reads **0.00%**. They retire in equal numbers. The opcode histogram of that
symbol reports `movzwl` at 33.98% (209.4 i/n) and `vpdpbusd` at 0.46% (2.8 i/n)
for a kernel whose entire purpose is `vpdpbusd`.

`instructions:pu` (precise) is **not available on this box** — `perf record`
exits 255 — so there is no cheap fix.

### 4.3 What this invalidates

* **Study 10 §3.2 / §0.3** — "the threat fold is at parity, 216.5 vs 219.7 i/n
  of `vpmovsxbw`", and the derived "SF folds ~6.9 threat rows per node to our
  ~5.4". The *conclusion* (parity) is re-established here by counting; the
  *numbers* and the row-ratio are artifacts.
* **Study 10 §3.2's opcode-class SIMD/scalar split** (SF 70/30 vs ARCHON 55/45)
  and every i/n figure derived from it, including "our threat SCALAR is ≈2.1×
  SF's, ≈450 i/n of excess" — the largest item study 10 left on the board.
* **Study 10 §5's row "HalfKA Finny refresh scalar half (6.8× SF's: 184 vs 27
  i/n)"**.
* Study 8/9 figures of the same construction were not re-derived here.

**Symbol-level shares are not affected** — skid stays within a symbol — so §3's
block table stands. It is only the *within-symbol opcode* attribution that
fails, which is exactly what the cross-engine eval comparisons were built on.

### 4.4 The replacement

For any workload question between two engines that compute the same features:
**count it**. Both patches are archived and re-runnable
(`counters11.patch`, `sfcount11-nnue_accumulator.cpp.txt`); the SF side took
one non-PGO build and about twenty lines.

---

## 5. IMPLEMENTATION — TWO ATTEMPTS, BOTH NULL

Probe pairs, native non-PGO, fixed 5,001,562-node tree. Raw: `probes.txt`.
**Every row has eval hash `5ca978e0…` and identical node count — both arms are
bit-exact and node-identical.**

| arm | instructions | per node | Δ | branches/node |
|---|---:|---:|---:|---:|
| `base-main` | 34,743,249,976 | 6946.5 | — | 843.1 |
| `psqt-rows-hoist` | 34,740,643,937 | **6946.0** | **−0.0075%** | 842.3 |
| `resolve-select` | 34,744,777,890 | **6946.8** | **+0.0044%** | **838.1** |

Both deltas are inside the instrument's own 0.0018–0.0034% spread. **Both are
nulls. Both were reverted** (`nulls11.patch`).

**§5.1 `psqt-rows-hoist`.** `psqt_row` called `table.as_chunks::<BUCKETS>()` on
every resolved event-perspective (~4.4/node) to reshape a table that is
invariant across the whole fold. Hoisting the reshape to `resolve_both` /
`resolve_into` should have removed a length-shift and a stack reload per event.
It removed 0.5 i/n. **LLVM was already hoisting it**; the disassembly's
per-event `cmp`/`jbe` is the *bounds check on the row index*, which the hoist
does not touch and which cannot be removed without giving LLVM a range fact for
`idx < THREAT_DIMS` (60720 is not a power of two, so study 10 §2.1's masking
trick does not transfer).

**§5.2 `resolve-select`.** The resolver's `if d.add { … } else { … }` gave each
arm its own store, cursor and PSQT accumulate; LLVM laid them out as two blocks
both jumping to a common tail, and three unconditional `jmp`s carried 8.5% of
`at_full`'s sampled weight. Replacing the branch with a selected destination —
`let bufs: [&mut …; 2] = [subs, adds]; bufs[d.add as usize][p][…] = idx` — plus
two same-sign PSQT partial sums combined as `add − sub` afterwards (bit-identical
for wrapping i32).

It worked as designed and bought nothing: **branches fell 843.1 → 838.1/node
(−0.6%) at +0.004% instructions.** The `cmov` on the base pointer costs what the
branch cost. Recorded because the 8.5% that motivated it was itself a §4
sampling artifact — **this is the second candidate in two studies to be sized
by that instrument and measure null.**

---

## 6. WHAT I DID NOT DO

* **No PGO deliverable binary of a changed state**, because nothing changed.
  The final tree is `main` at `51fe78e`, and §1.1 verifies it reproduces both
  NNUE-gate pins from source.
* **No game testing and no D33 review loop** — nothing landed to test or review.
* **The HalfKA fold was never counted deterministically.** Only threats were.
  SF's side is already collected (2.2585 psq delta rows/node); ours is one
  counter away, and study 10's Finny claim (§4.3) is now unsupported until it
  is done.
* **SF's `Network::evaluate` was never split deterministically.** Its 1335 i/n
  is a single symbol containing fold + transform + head. §3's eval comparison is
  therefore block-to-block, not stage-to-stage.
* **The ARCHON eval gate (study 10 §4.2, 157–732 i/n)** was not touched. It
  remains the largest uncosted item and is unaffected by anything here.
* **The copy-only fold** (0.3355/node, ~30 i/n) was designed and abandoned: it
  needs `at_full` to return per-perspective references rather than
  `&[[i16; ACC]; 2]`, which changes `activate_ft`'s and the SIMD kernels'
  signatures — too large to implement and gate safely in the time.
* `instructions:pu` was tried once and is unavailable; no other precise
  instrument was attempted.

---

## 7. RANKED REMAINING OPPORTUNITIES

| # | item | size | confidence | cheapest test |
|---|---|---:|---|---|
| 1 | **The head, `forward_into`** | 616 i/n | low — never opened by any study | counters for evals/node + nnz groups/eval, against the same in an instrumented SF (the `sfcount11` harness exists) |
| 2 | **HalfKA fold + Finny** (`current_full` + `finny_refresh_into`) | 873 i/n | low — study 10's 6.8× claim is retracted (§4.3) | one counter for HalfKA rows/node vs SF's already-measured 2.2585 |
| 3 | **ARCHON eval gate** (study 10 §4.2) | 157–732 i/n | medium, retrain-dependent | `evaldiff.py` restricted to `\|material_imbalance\| > 962` — fidelity, not speed |
| 4 | **make/unmake at 1.93×** | +243 i/n | medium — like-for-like per §3 | profile `make_move_with` against SF's `do_move` by phase |
| 5 | **Movepick at 1.16×** | +106 i/n | medium — contradicts study 8's parity finding vs SF18 | re-run study 8's movepick split against `eca43a97` |
| 6 | **Threat scalar plumbing** | ≤556 i/n, and two probes found nothing | **low** — §5 is two nulls, and its §4 sizing is retracted | count, don't sample: instrument `build_deltas`/`resolve_*` phases |
| 7 | Accumulator 1280 → 1024 | ~1.25× on all fold arithmetic | n/a | not an optimisation — an architecture decision needing a retrain |

**Item 6 is deliberately ranked last despite being study 10's recommendation
(its §5.1 item 2, "460 i/n, now the largest item that is genuinely ours").
That figure came from the retracted instrument, and two direct attempts on that
code this study returned null.**

---

## 8. THREATS TO VALIDITY

1. **§3 mixes frames** — ARCHON non-PGO, SF PGO. The eval block is bracketed
   3,155–3,355 i/n; ordering and signs are unaffected, absolute i/n are not.
2. **§0.3's 198 i/n is a net effect of three simultaneous differences** between
   SF18 and `eca43a97` (small net removed, threat set −24%, L2 +113%), not an
   isolation of the small-net term.
3. **§2's SF counters were added at two sites** (`apply_combined`'s caller and
   the refresh path's `append_active_indices`). If `eca43a97` folds threat rows
   anywhere else, SF's row count is understated — which would make our 0.88×
   look *better*, not worse, so the parity conclusion is robust to the error's
   direction.
4. **§2 compares one bench suite** (wl50, 30k nodes/position). Row counts are
   position-mix dependent; no second suite was run.
5. **Node counts differ by ~0.05%** between engines (15,001,568 vs 15,008,494)
   because the bench is node-limited per position, not tree-identical. This is
   inherited from studies 8–10 and is far below every effect discussed.
6. **Nothing here was Elo-tested**, and nothing here could be: no code changed.
7. **Box state**: load 7.5–9.6, 3–5 GB free, swap full, for the entire session.
   All claims ride on `instructions:u` on fixed-node trees and on deterministic
   counters, both contention-immune.

---

## 9. ARTIFACTS (`perf-artifacts-11/`)

* `counters-headline.txt` — the §0.1 three-arm panel, 6 rounds.
* `probes.txt` — every implementation arm with its eval hash and node count.
* `flat-{s11base,sfeca}-instructions.txt` — the §3 flat profiles.
* `annot-{at_full,forward_into}-s11.txt`, `annot-sfeca-evaluate.txt` — the
  annotations §4 retracts, kept as the evidence for the retraction.
* `counters11.patch` — ARCHON's §2 instrumentation (reverted).
* `perfcount11.rs.txt` — the counter module (removed).
* `sfcount11-nnue_accumulator.cpp.txt` — SF's §2 instrumentation.
* `nulls11.patch` — §5's two reverted implementations.
* `scripts/`
  * `probe11.sh` — the pricing pair, retargeted at this study.
  * `opsplit11.py` — **new**: opcode-class histogram of one symbol in absolute
    i/n. Use only with §4's caveat.
  * `opcount11.py` — **new**: whole-engine total for named opcodes across every
    symbol above 0.05%, which is what discharged study 10 §6.5's open
    assumption that `vpmovsxbw` appears in only one symbol (it does not — SF
    has it in three).
  * `pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`, `panicedge10.py` and the
    rest inherited from `perf-artifacts-10/scripts/`.

**Reproduction.**

```bash
# identity pins of the final tree (from source, sota,arch-d48x)
./target-probe11/release/chessengine nnue-eval engines/canonical/net.cenn \
  < tools/nnue/calib/wl50.fens | sha256sum          # -> 5ca978e0…0b38
./target-probe11/release/chessengine bench 16 1 13 \
  tools/nnue/calib/wl50.fens depth engines/canonical/net.cenn   # -> 1922947

# the headline panel
bash docs/research/perf-artifacts-11/scripts/pstat8.sh 6 300000 \
  tools/nnue/calib/wl50.fens /tmp/c.txt \
  "archon:./engines/bin/archon-s10-final @BENCH@ engines/canonical/net.cenn" \
  "sfeca:./engines/bin/stockfish-eca43a97 @BENCH@" \
  "sf18:./engines/bin/stockfish18 @BENCH@"
python3 docs/research/perf-artifacts-11/scripts/parse8.py /tmp/c.txt sfeca

# the target is single-net
./engines/bin/stockfish-eca43a97 uci | grep -ci small     # -> 0

# whole-engine opcode totals (and why they cannot be trusted — §4)
python3 docs/research/perf-artifacts-11/scripts/opcount11.py \
  docs/research/perf-artifacts-11/perf-sfeca-instructions.data 4452.4 vpmovsxbw

# the deterministic row counts (§2) — apply, build, run, revert
git apply docs/research/perf-artifacts-11/counters11.patch
```
