# ARCHON performance study #6 — post-optimization reprofile + independent verification

Date: 2026-07-23. The after-picture to study #5
(`perf-study-5-measured-2026-07-23.md`, the pre-change ground truth) and the
independent audit of the two optimization packages study #5 could not verify
(`perf-study-4-eval-2026-07-22.md` Appendix B = implementer claims).
Raw artifacts: `perf-artifacts-6/`.

**Source state measured:** main tip `280b646` (= merge `c6e820d` of the
optimization branch + merge-resolution `4c79c6b`), built in a synced worktree,
`RUSTFLAGS="-C target-cpu=native" cargo build --release --features sota`
(v3 build for callgrind only). Pre-optimization reference: the frozen
`engines/bin/archon-d47` (untouched; its IPC 2.27 and pinned NPS ≈ 608 k
reproduce study #5's baseline 2.26 / 592 k — it IS the 32e0a9c-era build).
Incumbent reference: `engines/bin/archon-sota-native` + `d40-uho-v1-240sb`,
deliberately untouched, measured as-is.

**Workload:** study #5's recipe — `bench 16 1 <depth> wl50 depth
data/nnue/nets/d47c-stage2-sb240.cenn` (.cpol auto-loaded), wl50 = every-16th
line (`NR%16==1`) of `tools/nnue/calib/scale-check-800.fens`
(`perf-artifacts-6/study6-wl50.fens`). **Subset caveat:** study #5's wl50 was
a different every-16th phase of the same (unchanged) file — their absolute
node totals (e.g. 1,874,757 at d12) are not reproducible from this subset
(mine: 1,486,769, and the frozen pre-opt binary produces exactly the same
count, proving it's the subset, not the tree). All old-vs-new comparisons
here use the identical subset on both sides; cross-study comparisons are
ratio- or per-event-based.

**Tripwires passed:** depth-6 default bench = 701605 (pin); depth-12
default-with-net fingerprint = 4,297,837 (Appendix B value, exact); PSQT
3.67 M NPS; with-net NPS in the 480–800 k band depending on FEN set (default
fens 480 k, wl50 ~800 k — wl50 is endgame-heavier).

**Box state:** NOT quiet — a cutechess SF16-vs-SF18 gauntlet (~2 floating
cores) ran throughout. All NPS numbers therefore come from study #5 §8's
interleaved round-robin (one run of every config per round, engine pinned
`taskset -c 4`, per-round ratios, median of 6 rounds); parity, node-identity,
callgrind, and perf-stat facts are contention-immune.

---

## 1. Verification verdicts — claimed vs measured, per package boundary

Chain rebuilt from `git archive` at each boundary commit, identical flags.
Parity oracle: 800-FEN `nnue-eval` byte-parity (cp AND policy-argmax columns,
sidecar active) + per-position node-count identity, wl50 × d13
(`perf-artifacts-6/study6-parity-hashes.txt`: one sha256 across all four
binaries for each oracle).

| boundary | package | claimed NPS | measured (median paired, 6 rounds) | bit-identical? | verdict |
|---|---|---:|---:|---|---|
| archon-d47 → a3f892e | copy-elim `9b695fb` + threat consts `a3f892e` | +3.7% / +0.3% (#5 §11) | **+3.1%** [+2.1..+5.0] | (verified in #5) | re-confirmed |
| a3f892e → b46c026 | **head kernels** `b46c026` | **+21.4%** | **+22.8%** [+20.5..+23.7] | **YES — 800/800 eval, 50/50 nodes** | **CONFIRMED (slightly undersold)** |
| b46c026 → tip | **policy hygiene** `3555f57` | **+2.2%** | **+3.8%** [+1.0..+6.4] | **YES — 800/800 eval, 50/50 nodes** | **CONFIRMED (undersold)** |
| archon-d47 → tip | end-to-end | ~+29% | **+30.9%** [+29.0..+34.4] | YES (composition) | **CONFIRMED** |

Both previously-unverified packages are real, bit-exact, and worth slightly
more on the pinned workload than the implementer claimed. The de-Box and
copy-elimination claims are also count-verified (§4): malloc on the search
path went from 1.015/eval to **zero** (4,147 total calls per run = startup
only), memcpy from 9.4/eval to 2.80/eval.

## 2. The new hot-path table (perf cycles, native, wl50/d16, 56,009 samples)

The native build now statically dispatches AVX-512 and inlines every head
kernel into `SotaHead::forward_into`, and the fold kernels into their
drivers — symbols merged accordingly (mapping in col 4).

| # | symbol (new binary) | cycles % | contains (study-#5 symbols → their % sum) |
|---|---|---:|---|
| 1 | `SotaHead::forward_into` | **37.0** | l1_dot 34.4 + forward_full 7.6 + activate 5.9 + find_nnz 0.5 = **48.4** |
| 2 | `ThreatContext::at` | 11.4 | at 2.5 + i8_deltas_into 7.5 = 10.0 |
| 3 | `HalfkaContext::finny_refresh_into` | 8.8 | finny_refresh 2.0 + i16_deltas share 3.9 + finny memcpys ~1.0 |
| 4 | `MovePicker::next_move` | 7.0 | 4.9 |
| 5 | `HalfkaContext::current` | 6.8 | current 1.8 + i16_deltas_into 4.7 = 6.5 |
| 6 | `Board::make_move_with` | 4.6 | 1.2 (+ inlining shifts) |
| 7 | `Worker::search::<NonPv>` | 3.7 | 3.1 |
| 8 | `MovePicker::new` (+probcut 0.3) | 2.2 | 1.9 |
| 9 | `sota_threats::record_square_touch` | 2.0 | 1.5 |
| 10 | `SotaContext::evaluate` (driver) | 1.7 | — |
| … | `PolicyHead::best_move` | **0.36** | 1.53 (vectorized policy_dot: ~4× cheaper) |
| … | `TranspositionTable::probe` | 0.20 | 0.9 |

Within-symbol attribution of forward_into (perf-annotate region split,
`perf-artifacts-6/annotate-forward_into.txt`):

- **l1_dot loop ≈ 68% of the symbol ≈ 25% of ALL cycles** — and it is now the
  *designed* shape: one nnz walk, 8 × `vpdpbusd` into 8 distinct zmm
  accumulators, zero bounds-check edges. The hot spots are the 8 weight loads
  and the loop-carried compare (19.6% on `cmp` = the skid sink for the
  load-port stalls). This is the mca-predicted load-port-ceiling regime.
- f32 L2 tail ≈ 10% of symbol ≈ 3.6% of cycles — now true zmm
  `vfmadd231ps`/`vbroadcastss` (the 2-of-16-lane xmm defect of #5 §2b is
  dead).
- FT split-quant merge + glue ≈ 19% of symbol ≈ ~7% of cycles
  (`vpmovsxwd`/`vpsrld`/`vpmovdb` region) — the THREAT_QUANT≠FT_QUANT tax,
  still alive, still a format-rev item.
- find_nnz ≈ 3% of symbol; l1_dequant ≈ noise.

### Where did l1_dot's 34.4% go?

In absolute terms: 34.4% × 60.4 G cycles ≈ 20.8 G → ~25% × 46.8 G ≈ 11.7 G
cycles on the same tree — **the l1_dot loop itself got ~1.8× faster** (mca's
3.0× kernel bound minus the memory reality: the ~100 KB/eval L1-weight
stream still misses L1d at ~25%). The rest of the head (forward_full's
xmm tail, activate copies, dequant) shed another ~6 points. **The new #1 is
still the head forward pass — but it is now bytes-bound, not
wasted-instruction-bound.** What remains in l1_dot is density-linear load
traffic, which no kernel change can remove (see §6 item 1).

### Block totals

| block | #5 (pre) | #6 (post) | note |
|---|---:|---:|---|
| head (forward_into + driver + push) | 48.5% | **≈ 39.1%** | absolute cycles −43% |
| HalfKA + finny | 12.4% | ≈ 15.6% | share up, absolute ≈ −2% (untouched work) |
| threat channel | 12.4% | ≈ 14.5% | share up, absolute −9% (fewer folds, §4) |
| policy machinery | ~2–3.5% | **≈ 0.4%** | consult 4× cheaper + stash free |
| **eval layer total** | ≈ 76% | **≈ 70%** | |
| search machinery | ≈ 24% | **≈ 30%** | now the second-largest "block" |

## 3. Counter rates — identical 6,852,134-node tree (wl50/d15, pinned)

| metric | archon-d47 (pre) | tip (post) | delta |
|---|---:|---:|---|
| NPS (this run) | 610,000 | 794,266 | **+30.2%** |
| cycles | 60.44 G | 46.82 G | **−22.5%** |
| instructions | 137.08 G | 83.68 G | **−38.9%** |
| IPC | 2.27 | 1.79 | −21% |
| branches | 19.76 G | 10.79 G | **−45.4%** |
| branch-miss rate | 1.0% | 1.6% | abs misses −11% |
| L1d loads | 57.41 G | 41.43 G | −27.8% |
| L1d miss rate | 26.0% | 25.2% | ~unchanged (the weight stream) |
| L1d misses abs | 14.92 G | 10.46 G | −29.9% |
| dTLB misses | 13.87 M | 13.65 M | unchanged |

Reading: the optimization removed *instructions* (bounds checks, 8× nnz
re-walks, scalar/xmm tails — hence −39% instr, −45% branches) and *repeat
traffic* (single-pass weight walk — hence −30% L1d misses), while IPC fell
because the surviving stream is denser in real loads. The engine is now at
the 1×512-bit-load/cycle port ceiling inside l1_dot; #5's L2-resident
verdict stands (DRAM was 0.33 GB/s pre-change and the byte volume only went
down). Further kernel-level ILP is exhausted — remaining levers are
algorithmic (fewer bytes) or structural (quant unification).

## 4. Deterministic schedule counts (callgrind, v3→AVX2 path, wl50/d12)

1,486,769 nodes, 614,542 full evals = 0.413 evals/node. Total 39.32 G Ir
(26.4 k Ir/node vs #5's 30.6 k on their subset — AVX2-path Ir −14%; the
native-path instruction cut is the −39% above; Ir ≠ cycles, the AVX2
kernel's win is accumulator ILP, not instruction count).

| event | calls | per eval | #5 (their subset) | reading |
|---|---:|---:|---:|---|
| threat folds `i8_deltas_into` | 1,174,215 | **1.91** | 2.70 | **−29%: the degenerate/null-move folds are aliased, not copied (S2 landed)** |
| HalfKA folds `i16_deltas_into` | 989,311 | 1.61 | 1.72 | unchanged (subset) |
| finny refreshes | 444,965 | 0.72 | 0.67 | in-place now (no 4-memcpy round-trips) |
| threat pushes `apply_at` | 1,489,932 | 1.00/node | 1.00/node | eager remnant, still trivial |
| policy consults `best_move` | 15,050 | 2.4% of evals | 2.3% | gate unchanged |
| `policy_dot` rows | 346,308 | 23.0/consult | scalar loop | legal-rows-only, vectorized |
| **malloc / free** | **4,147 / 4,135** | **~0** | 1.015/eval | **de-Box CONFIRMED: search path is allocation-free again** |
| memcpy | 1,719,915 | **2.80** | 9.4 | copy elimination −70% |

## 5. Deployment ladder (median of 6 interleaved pinned rounds, wl50/d14)

| config | median NPS | vs ship config |
|---|---:|---:|
| PSQT (tip, no net) | 3,422,783 | 4.30× |
| **incumbent: archon-sota-native + d40-uho-v1-240sb** | **1,238,641** | **1.556×** |
| stage2-nocpol (sidecar hidden) | 811,033 | +1.5% |
| **stage2+cpol (ship config, tip)** | **796,730** | — |
| stage1+cpol (d47c-sb240) | 758,350 | −4.6% |
| frozen pre-opt (archon-d47, stage2+cpol) | 607,806 | −23.7% |

- **Equal-time bar math for the crown rematch:** the d47c challenger's speed
  deficit vs the incumbent fell from **2.04×** (pre-opt: 1,238,641/607,806)
  to **1.56×** — from 1.03 to 0.64 time-doublings. At the usual 70–90
  Elo/doubling for these TCs, the fixed-node Elo advantage the d47c net must
  show to break even at equal time dropped from ≈ **72–92 Elo** to ≈
  **45–58 Elo**. The optimization bought back roughly 40% of the width tax.
- **The sidecar cycle tax collapsed: 3.5% → 1.5%** (vectorized policy_dot +
  de-Box + free stash). The #5 §5 tree-inflation finding (+6.2% nodes at
  equal depth) is a search effect these bit-identical patches cannot touch —
  combined cost while the head buys 0 Elo is now ~8%, nearly all of it tree.
- **stage1 is 4.6% slower than stage2 at identical arch and positions** —
  direct evidence that FT activation density is a per-net property that
  moves NPS (see item 1 below). The net tax vs PSQT fell 5.54× → 4.30×.
- Study #4 Appendix B's absolute ladder (495 k/540 k/534 k, bench-d12
  default fens) sits below these numbers because the default-fens workload
  is opening/middlegame-heavy; ratios agree directionally except the sidecar
  tax, which their default-fens run put at 8–9% — on wl50 it is 1.5%;
  fire-rate and legal-move counts are position-mix-dependent. Quote the
  workload with any sidecar-tax number.

## 6. Next five, ranked

1. **FT activation-sparsity as a trainer-side lever (next retrain).** l1_dot
   is still the single largest cost (~25% of cycles) and is now clean — what
   remains is density-linear weight-load traffic (#5 measured ~62% group
   density = ~100 KB/eval vs the 768-era ~10–15%). No inference-side trick
   is left; the lever is making the FT sparse again in training (L1
   activation penalty / SCReLU-knee shaping on the next d47c-class run).
   62%→30% density ≈ halves l1_dot ≈ **+12–14% NPS at zero inference risk**,
   and the stage1-vs-stage2 4.6% NPS gap proves density varies across
   otherwise-identical nets today. Action now: add an avg-nnz probe to the
   net gate so every candidate reports its density before deployment.

2. **Search-lane micro-pass.** The eval layer's deflation makes search ~30%
   of cycles: `next_move` 7.0% + `MovePicker::new(+probcut)` 2.5% +
   `make_move_with` 4.6% are the largest untouched items in the engine.
   Study #5's S3 (movelist-state memcpy per picker construction) plus the
   staged-selection shape of next_move have never had a dedicated pass.
   Target **+3–5% NPS**; ordering-neutral parts are node-count-verifiable,
   anything touching selection order goes through SPRT.

3. **Sidecar consult: decide on Elo, not NPS.** With the cycle tax at 1.5%,
   the remaining cost is the +6.2% equal-depth tree inflation — a *search*
   effect frozen into the gate, not the kernels. Re-run the ON/OFF ablation
   with the current sb240 head at 10+0.1; if the consult still buys ≤ 0 Elo,
   add the trivial env/UCI gate to ship consult-OFF while keeping the
   sidecar loaded for probes (**~8% effective speed** reclaimed for free
   until a head actually wins its ablation).

4. **T10 smallnet — keep parked behind the in-tree histogram.** The case
   weakened again: position-weighted p is 0.08–0.15 (#5 §6), the eval it
   would bypass costs 22.5% fewer cycles, and the PSQT multiple fell
   5.54×→4.30×, shrinking the max bypass win. Revised expectation at the
   8–15% fire band: **+5–8% NPS** for HIGH effort plus an accuracy/Elo risk.
   The cheap next step stands: a 20-line in-tree imbalance counter at the
   real gate; train only if tree-weighted p ≥ ~0.15.

5. **Quant unification at the next format rev.** The FT split-quant merge
   (THREAT_QUANT ≠ FT_QUANT) survives as ~7% of cycles of vpmovsxwd/vpsrld/
   vpmovdb glue inside forward_into, and it forces the u8 FT round-trip
   through the stack. Folding threat+halfka into one quant domain in the
   next .cenn rev deletes the merge and lets l1_dot read one pre-activated
   buffer. NOT bit-exact — ride the next net-format change with its own
   parity oracle. Riders while in there: bulk net parse (read_i8s/read_i16s
   still ~0.4–0.5 s per spawn = SPRT throughput), find_nnz VBMI2 already
   landed; THP stays demoted (#5 B5); prefetch stays dead everywhere it was
   tried (#4 App-B).

## Artifacts (`perf-artifacts-6/`)

- `perf-report-tip{,-full}.txt`, `perf-tip-wl50d16.data`,
  `annotate-forward_into.txt` — the new cycle profile (native).
- `callgrind-tip-wl50d12.out`, `callgrind-annotate-tip.txt` — deterministic
  Ir + call counts (v3/AVX2 path).
- `perfstat-{d47,tip}-wl50d15.txt` — the identical-tree counter pair.
- `study6-ladder-raw.txt` — all 48 interleaved ladder cells.
- `study6-parity-hashes.txt`, `study6-wl50.fens` — parity oracle hashes +
  the exact FEN subset (regen: `awk 'NR%16==1' tools/nnue/calib/scale-check-800.fens`).
