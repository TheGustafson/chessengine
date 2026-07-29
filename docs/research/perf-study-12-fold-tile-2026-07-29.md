# ARCHON perf study #12 — the fold kernels, and five things that landed

Date: 2026-07-29. Twelfth commissioned study (8: `perf-study-8-nps-parity-2026-07-28.md`,
9: `…-9-parity-items…`, 10: `…-10-parity-floor…`, 11: `…-11-fair-target-2026-07-29.md`).
Raw artifacts and instruments: `perf-artifacts-12/`. Baseline: `main` at `29cd9cb`
(study 11's merge plus its framing correction). Comparison target:
**`engines/bin/stockfish-eca43a97`**.

---

## 0. HEADLINE

### 0.1 The ratio moved

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, three arms interleaved,
`taskset -c 4`, `nice -n 19`, 4 rounds, all arms PGO. Raw:
`counters-headline.txt`.

| per node | **ARCHON study 12** | ARCHON study 11 | **SF `eca43a97`** |
|---|---:|---:|---:|
| **instructions** | **6113.5** | 6367.1 | **4452.4** |
| **ratio vs `eca43a97`** | **1.3731×** | 1.4301× | — |
| branches | 667.5 | 734.2 | 362.7 |
| branch-misses | 20.6 | 22.2 | 19.7 |
| paired NPS ratio | 0.6673 | 0.6425 | — |
| `instructions:u` spread | 0.0389% | 0.0034% | 0.0000% |

The instrument is validated by its own controls, twice: **SF reproduces 4452.4
exactly**, and study 11's own binary reproduces **6367.1 against its published
6367.2**.

**1.4301× → 1.3731×. −253.6 i/n, 13.2% of the 1,914.7 i/n gap, in five
node-identical changes.** Branches fell 9.1%.

### 0.2 What landed

Every arm is **node-identical** (5,001,562 on the probe tree; 15,001,568 on the
headline bench) and **eval-hash-identical** (`5ca978e0…0b38`). The pricing
instrument is a fixed-tree `instructions:u` probe whose own null band is
**±0.15 i/n over three reruns of identical source** (§1.1).

| # | change | probe Δ | where |
|---|---|---:|---|
| 1 | AVX-512 `_into` fold kernels: drop the `feature * ACC` offset hoist + CAP chunking | **−114.1** | `sota_simd.rs` |
| 2 | movepick offense term as a reverse attack fan | **−33.8** | `movepick.rs` |
| 3 | HalfKA backward-fill of the span below a king move | **−2.5** | `sota_accumulator.rs` |
| 4 | **AVX-512 fold tile 8 → 20 registers** | **−314.4** | `sota_simd.rs` |
| 5 | threat generator: reuse the ray fans instead of re-running the magic lookup | **−44.4** | `sota_threats.rs` |
| | **total, probe frame** | **−509.2** | |
| | **total, PGO frame (the number that counts)** | **−253.6** | |

### 0.3 The methodological correction this study owes the next one

**The non-PGO probe overstates these wins by ~2×.** −509.2 i/n of probe delta
became −253.6 i/n of PGO delta, because PGO was already recovering roughly half
of each defect on its own. The probe is still the right *screening* instrument —
it is exact, its null is ±0.15 i/n, and it costs one minute — but **no probe
delta may be quoted as a result.** Only the PGO panel decides.

---

## 1. METHOD

Studies 8–11's rules were followed, plus study 11's replacement rule ("count it,
do not sample it"). Three instruments:

* **`probe12.sh`** — pinned release profile (`lto = "fat"`, `codegen-units = 1`),
  native, non-PGO, fixed 5,001,562-node tree. Prints the wl50 eval hash and node
  count so a tree or eval change cannot hide inside a speed claim.
* **`probe12fast.sh`** — the same instrument at `CARGO_PROFILE_RELEASE_LTO=thin`
  / `CODEGEN_UNITS=4`. ~20× faster to build and a fraction of the peak RSS, which
  is what made a dozen arms affordable. Absolute i/n is not comparable to the
  pinned frame; only fast-to-fast deltas are.
* **`pgo12.sh` + `pstat8.sh`** — the deliverable frame.

### 1.1 NULL FIRST

Before any arm, the same source was built and measured three times:

```
base-main      34,744,783,338   6946.8 i/n
null-rerun-1   34,743,796,494   6946.6 i/n
null-rerun-2   34,743,279,003   6946.5 i/n
```

Spread **0.0043%, ±0.15 i/n**. The fast frame's null is tighter still
(6895.9 twice, 0.00007%). Every landed arm is 15× to 2000× the null.

### 1.2 Gate status

`CHECK_NNUE_GATE=1 bash tools/check.sh` — see §9.

---

## 2. THE COUNTER PANEL — both engines, same bench

Study 11 left "the HalfKA fold was never counted deterministically" as its open
item #2. It is counted now, on **both** engines
(`counters.txt`, `counters-r2.txt`; ARCHON patch `counters12.patch`, SF patch in
`/home/nick/sfcount12`). Row counts are workload quantities: immune to codegen,
PGO, sampling and box load.

| per node | **ARCHON (baseline)** | **SF `eca43a97`** | ratio |
|---|---:|---:|---:|
| **evals / node** | **0.4315** | **0.4412** | 0.978 |
| nnz groups / eval | 98.98 | 104.19 | 0.950 |
| … out of how many groups | 320 | 256 | 1.250 |
| HalfKA delta rows | 1.6556 | 2.2585 | 0.733 |
| HalfKA refresh calls | 0.3167 | 0.1673 | **1.893** |
| HalfKA refresh rows | 1.5183 | 0.9268 | 1.638 |
| HalfKA rows, total | 3.1739 | 3.1853 | 0.996 |
| threat delta rows | 4.0481 | 3.8005 | 1.065 |
| threat refresh rows | 0.1609 | 0.9626 | **0.167** |
| **all accumulator rows** | **7.3828** | **7.9484** | **0.929** |

**Two facts settle most of the open questions from studies 8–11.**

**(a) `evals/node` are the same.** 0.4315 vs 0.4412. So the eval-block ratio is a
genuine per-eval ratio and not a call-rate artifact — nobody had checked.

**(b) We already fold FEWER rows than Stockfish**, 0.929×, across *both* feature
sets. The eval gap is therefore **not row count**. It is what we spend per row
and per pass.

---

## 3. THE FINDING: passes, not rows

The quantity that separates the two engines is **accumulator passes per node** —
how many times a full accumulator is read and written:

| | ARCHON | SF |
|---|---:|---:|
| HalfKA passes / node | 1.1020 | — |
| threat passes / node | 1.2005 | — |
| **total** | **2.3025** | **1.1655** |
| × lanes (1280 vs 1024) | 2947 | 1194 |
| **lane-copies / node** | | **2.47×** |

We keep the HalfKA and FullThreats accumulators **separate**; Stockfish folds
both feature sets into one array in one `apply_combined` pass. That is the single
largest structural difference in the eval layer.

**It is nonetheless the right design, and this study priced it rather than
assuming.** Merging would save the second pass (~109 i/n) and simplify
`activate` from 11 to ~7 instructions per 32 products (~69 i/n) — but it would
force the threat channel onto HalfKA's refresh trigger, and our lazy
king-cross scheme is the reason we pay **0.1609 threat refresh rows per node
against Stockfish's 0.9626**. Restoring SF's rate costs ~64 i/n back. Net ~114
i/n for a rewrite of three modules. **Do not merge the accumulators** — §8 item 6.

---

## 4. LANDED #1 — the offset hoist that was in the wrong backend (−114.1)

`i16_deltas_into_inner` / `i8_deltas_into_inner` (AVX-512) hoisted
`feature * ACC` into a `[MaybeUninit<usize>; 96]` stack array, chunking the
feature lists to fit it.

Disassembly of the executed path: **174 instructions before the tile loop is
even reached**, of which ~91 execute — a 1656-byte stack frame, six callee-saved
pushes, a CAP `min`/`cmov` pair per list, and, because the trip count is a
runtime value, LLVM's fully generic vectorized offset loop (a `vpmullq` 8-wide
body plus its scalar remainder) **emitted twice**. The measured trip count of
that loop is **one**.

The hoist is correct — *in the AVX2 backend it was written for*. That nest is
block-outer, `ACC / LANES` = 40 iterations, so it saves 39 × rows multiplies.
The AVX-512 nest holds a register tile and runs its outer loop 5 times (now 2),
so it saved 4 × rows. It was copied across; the prologue dominated. **The AVX2
copy keeps the hoist and now carries a comment saying why.**

Bit-identical: same rows, same per-lane wrapping i16 adds, same order. Dropping
the chunking is bit-identical too — mod-2^16 addition is associative.

---

## 5. LANDED #2 — the offense term, turned inside out (−33.8)

`init_quiets` scored the D30 "offense" term with **one magic-bitboard lookup per
scored quiet**:

```rust
let att = att_tables.attacks(pt, m.to(), occ);
if !(att & vulnerable[pt]).is_empty() { value += … }
```

Stockfish's `score<QUIETS>` has no analogue at all — this is an ARCHON original.
But the attack relation is **symmetric under a fixed occupancy**: `sq ∈
attacks(pt, to, occ)` iff `to ∈ attacks(pt, sq, occ)`, because both statements
say "sq and to lie on a pt-ray and every square strictly between them is empty in
occ". So the union of the *reverse* fans over the vulnerable enemy squares is one
bitboard per piece type, and the per-move test becomes one `contains`.

Lookups drop from ~one per quiet (~30) to `Σ_pt popcount(vulnerable[pt])` — and
those sets are nested and usually near-empty. Pawns are the one asymmetric case:
the reverse fan flips colour.

Node-identical ⇒ the ordering is provably unchanged.

---

## 6. LANDED #3 — the backward fill Stockfish does and we did not (−2.5)

Our `HalfkaContext::compute` refreshed the leaf and marked **only the leaf**
computed. Every ply between the king move and that leaf stayed uncomputed, so the
next sibling below any of them walked its ancestor search straight past the king
move and refreshed from scratch again.

Stockfish's `AccumulatorStack::evaluate` (`eca43a97:src/nnue/nnue_accumulator.cpp`)
refreshes at the latest state and then runs
`backward_update_incremental` down to the refresh-requiring ply, which **marks
the intermediates computed**. `find_last_usable_accumulator` is what makes the
siblings cheap.

The refresh *trigger* is identical in both engines
(`HalfKAv2_hm::requires_refresh` is "this perspective's king moved", exactly
ours), so the 1.893× refresh-rate ratio was purely this.

Implemented as `apply_dirty_directed::<FORWARD>` — the backward direction swaps
the add and sub lists, which is exact because the fold is a sum of whole rows and
wrapping i16 addition is invertible. The current king square is valid for the
whole backward span precisely because the span *starts* at the king move.

Measured after (`counters-r2.txt`):

| per node | before | after | SF |
|---|---:|---:|---:|
| finny refreshes | 0.3167 | **0.1611** | 0.1673 |
| finny refresh rows | 1.5183 | **0.9064** | 0.9268 |
| reverse folds (new cost) | — | 0.0532 | — |

**The rate is now marginally better than the engine the idea came from.** The
instruction win is small (−2.5 i/n) because a refresh and two reverse folds cost
about the same; the win is the mechanism, and it removes 57 MB of random weight
traffic per 1000 nodes.

---

## 7. LANDED #4 — the register tile (−314.4, the study's largest single win)

The fold kernels' *arithmetic* is fixed at `rows × BLOCKS` load/add pairs. Their
*loop control* is `tiles × rows × ~6` — load the feature, `lea`+`shl` for
`feature * ACC`, bump, compare, branch. **It scales with the number of tiles, not
with the work.**

Study 2 chose `REGS = 8`: five tiles at ACC = 1280, spending ~220 i/n of control
on ~7.1 rows per node. AVX-512 has 32 zmm registers and the body needs exactly
one live register beyond the accumulators. `REGS = 20` gives **two** tiles at
1280 and leaves 11 registers spare.

```
REGS = 8    6895.9 i/n   (5 tiles)
REGS = 20   6581.5 i/n   (2 tiles)   -314.4
REGS = 40   7589.2 i/n   (1 tile)    +1007.7   <- 40 accumulators spill
```

`pick_regs()` takes the largest divisor of `BLOCKS` in `[9, 20]`, so the
remainder never falls into the one-block tail loop: 1280 → 20, 768 → 12,
1536 → 16, 2048 → 16. The resulting kernel is what you would write by hand —
20 `vmovdqu64` loads, then 20 `vpaddw (mem)` per row with the row base as
`lea (rbx,rbx,4)` / `shl $9`, then 20 stores, with the tile loop fully unrolled
into zmm0–zmm19.

---

## 8. LANDED #5 — the ray fans computed twice (−44.4)

`record_square_touch` computed the mover's outgoing attack set with
`tab.attacks(pt, s, occ)` and then, a few lines later, computed `ratt` and `batt`
at the *same square with the same occupancy* for the x-ray ripple. For a rook or
bishop that is the identical magic lookup twice; for a queen, `attacks(QUEEN,…)`
runs both lookups and then `ratt`/`batt` run them again — **four lookups where
two suffice**.

Hoisting `ratt`/`batt`/`qatt` above the match and dispatching the outgoing set
from them is bit-identical: `AttackTables::attacks` *is* `rook_attacks` /
`bishop_attacks` / their union for R/B/Q.

---

## 9. GATE

See the run log in `perf-artifacts-12/`. Both pins are unchanged, which is the
point: every arm is node-identical and eval-bit-identical.

* `Bench: 587683` — the default-feature depth-6 fingerprint. Arm #2 is in
  `movepick.rs`, which the default build *does* compile, so this is a real check
  on it.
* NNUE gate: eval hash `5ca978e0…0b38`, d13 nodes `1922947`.
* `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `cargo test`
  (679 lib tests; the per-node accumulator oracle `assert_matches_refresh` runs
  at every node of every debug search and validates arm #3), release perft suite.

---

## 10. WHAT THE PROFILE SAYS NOW, AND WHAT IT LIES ABOUT

PGO flat profile of the final binary (`flat-s12final-pgo-instructions.txt`,
6114.6 i/n). ARCHON blocks against SF's PGO profile
(`perf-artifacts-11/flat-sfeca-instructions.txt`, 4452.4 i/n):

| block | ARCHON | SF | excess |
|---|---:|---:|---:|
| eval layer | ~2976 | 1975 | **+1001** |
| search + histories + see | ~2440 | ~1945 | **+495** |
| make / unmake | 417 | 262 | **+155** |
| **engine** | **6113.5** | **4452.4** | **+1661** |

**Two corrections to study 11's table, both from measuring in one frame instead
of two:**

* **Movepick is at parity, not 1.16×.** Study 11 profiled ARCHON non-PGO against
  SF PGO. In one frame: ARCHON movepick 692 vs SF 675 at baseline, and after
  arm #2 we are *below* Stockfish.
* **`forward_into` is at parity too, and study 11's item #1 is closed.** A
  `__attribute__((noinline))` split of Stockfish's own eval
  (`flat-sfeca-split-instructions.txt`) prices SF's `transform` (activate + nnz)
  at 121 i/n and its head `propagate` at 451 — **572 total against our
  `forward_into` at 556.** Our L2 = 16 against their 32 pays for our 1.25× wider
  accumulator. **The head is not where the gap is; do not open it again.**

**And a warning about the instrument.** For the fold kernels, the flat profile
reads **2.2–2.5× the kernel's provable instruction count** (the disassembly is
20 loads + `rows × 20` `vpaddw` + 20 stores, and the counters give the rows).
SF's `apply_combined` measures *at* its arithmetic floor. The difference is
memory stalls: our weight tables are 57 MB (HalfKA i16) and 78 MB (threats i8),
every row is a guaranteed miss, and non-precise sampling piles samples onto
stalling instructions. `instructions:pu` is unavailable on this box (study 11
§4.2). **Treat flat-profile i/n for the fold kernels as an upper bound; price
them with the probe.**

---

## 11. RANKED REMAINDER — handoff to study 13

1,661 i/n remain. **~327 of it is the 1280 vs 1024 accumulator width**, which is
a retrain, not a code change (D48X is 1280). The rest is code.

| # | item | size | confidence | cheapest next test |
|---|---|---:|---|---|
| 1 | **`search::<NonPv>` 1251 vs SF 875** | +376 | low — never split | `#[inline(never)]` split of the search body's phases (the fast probe makes this a 1-minute build), then compare against SF's `search<NonPv>` split the same way |
| 2 | **make/unmake 417 vs 262** | +155 | medium | the `states.push(StateInfo::EMPTY)` constant store is **provably dead** — 3 zmm loads + 3 zmm stores of 168 B per node, confirmed in the disassembly, and LLVM will not DSE it across `set_check_info`. Needs `states` to become a pre-sized array with a `top` index instead of a `Vec` push/pop (D8 change) |
| 3 | **`at_full` 954** | ≤ +400 | medium | the threat channel's fold is inlined here; split it with `#[inline(never)]` and price the generator/resolver separately. Note §10's warning before believing any of it |
| 4 | Fold-kernel loop control, second cut | ~44 | medium | unroll the row loop by 2 (halves the remaining `tiles × rows × 6`) |
| 5 | `record_mutate`'s duplicated fans | ~5 | high | same fix as arm #5, one level up: `record_outgoing` × 2 + `record_incoming_mutate` recompute `ratt`/`batt` at the same square up to three times |
| 6 | ~~Merge the two accumulators~~ | **+114, then −64** | **priced and REJECTED** | §3. Merging costs our 0.167× threat refresh rate. Do not spend a session on it |
| 7 | ~~The head (`forward_into`)~~ | **0** | **closed** | §10 — 556 vs Stockfish's 572, measured |
| 8 | Accumulator width 1280 → 1024 | ~327 | n/a | a retrain, not an optimisation |

**Honest arithmetic on the 1.1× bar.** 1.1× is 4897.6 i/n; we are at 6113.5, so
**−1215.9 remains, of which ~889 is code** once the width term is set aside.
Items 1–5 above are the whole of that budget, and item 1 (never opened) is 42% of
it. It is reachable only if the search block turns out to carry a defect of the
same class as §4 and §7 — which is exactly what nobody has looked for yet.

---

## 12. THREATS TO VALIDITY

1. **Probe deltas overstate PGO deltas by ~2×** (§0.3). The per-arm figures in
   §0.2 are probe-frame; only the −253.6 total is PGO-measured. Per-arm PGO
   attribution was not run (5 extra PGO builds on a 15 GB box).
2. **§10's ARCHON/SF block table still mixes two PGO builds** of different
   compilers; block *ordering* is safe, absolute i/n are ±.
3. **The flat profile over-attributes the fold kernels** by 2.2–2.5× (§10).
   Every conclusion that depended on it was re-derived from the probe or from
   counters.
4. **Nothing here was Elo-tested.** Every arm is node-identical and
   eval-bit-identical, so there is nothing to test: the engine plays exactly the
   same moves, faster. Arm #3 changes accumulator *bookkeeping* only and is
   covered by the per-node debug oracle.
5. **Counters were taken at the baseline and again after arms 1–4**; arm #5
   post-dates the second panel and does not change any row count (it removes
   duplicate lookups, not events).
6. **Box state**: a D48Y training run shared the box for the first half of the
   session and was lost to the global OOM killer at 00:59 (a build of mine at
   3.9 GB RSS was the trigger). Everything after that ran with the box otherwise
   idle. All claims ride on `instructions:u` on fixed-node trees and on
   deterministic counters, both contention-immune.

---

## 13. ARTIFACTS (`perf-artifacts-12/`)

* `counters-headline.txt` — the §0.1 three-arm PGO panel, 4 rounds.
* `probes.txt`, `probes-fast.txt` — every arm with its eval hash and node count.
* `counters.txt` — the §2 dual-engine deterministic panel.
* `counters-r2.txt` — the post-change panel (§6's refresh rates).
* `counters12.patch`, `counters12-r2.patch` — the ARCHON instrumentation (reverted).
* `stagesplit12.patch` — the `#[inline(never)]` stage split (reverted).
* `flat-s12base-pgo-…`, `flat-s12final-pgo-…` — before/after PGO profiles.
* `flat-s12split…`, `flat-s12fast…` — the stage splits.
* `flat-sfeca-split-instructions.txt` — **Stockfish's own eval, stage-split**,
  the evidence for §10's "the head is at parity".
* `scripts/probe12.sh`, `scripts/probe12fast.sh`, `scripts/pgo12.sh` — new.
* `scripts/pstat8.sh`, `parse8.py`, `flat8.sh`, `annot8.py`, `opcount11.py` —
  inherited.

**Reproduction.**

```bash
# the headline panel
bash docs/research/perf-artifacts-12/scripts/pgo12.sh archon-s12final
bash docs/research/perf-artifacts-12/scripts/pstat8.sh 4 300000 \
  tools/nnue/calib/wl50.fens /tmp/c.txt \
  "archon12:./bin12/archon-s12final @BENCH@ engines/canonical/net.cenn" \
  "sfeca:./engines/bin/stockfish-eca43a97 @BENCH@"
python3 docs/research/perf-artifacts-12/scripts/parse8.py /tmp/c.txt sfeca

# one arm, screened
bash docs/research/perf-artifacts-12/scripts/probe12fast.sh <label>

# Stockfish's eval, stage-split (scratch tree, not the reference clone)
#   see §10; patches noinline transform/propagate/apply_combined in
#   /home/nick/sfcount12 and rebuilds with ARCH=x86-64-avx512icl
```
