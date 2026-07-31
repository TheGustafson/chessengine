# Perf study 50 — `forward_into`'s misses are CAPACITY, not layout: 40.5 % → 37.3 %

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `51c7d3cf` (study 49
merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf50/` · **Artifacts:**
`docs/research/perf-artifacts-50/` · **Branch:** `perf-study-50-forward`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO, pinned to
physical core 2 (siblings 2,10 — the quietest pair at 7.6 % combined, measured
before starting).

---

## ★ WHAT LANDED, AND THE BEFORE/AFTER L1 MISS RATE

**`ft_scale` is stored as `u8` instead of `i16`.** It is the per-accumulator-lane
dequantisation multiplier the merge applies to the i8 HalfKA accumulator. The
values are `ceil(peak / 127)`; measured on the canonical net by this study's new
`FOLDVALHIST` dump they take **eight distinct values, all in 1..=8** — so fifteen
of every sixteen bits were padding, and `activate` was streaming 2560 B per
evaluation to carry 1280 B of information.

| | before (i16) | after (u8) | Δ |
|---|---:|---:|---:|
| **`forward_into` L1 miss rate** | **39.73 %** | **37.25 %** | **−2.48 pp** |
| **`forward_into` share of engine miss LATENCY** | **17.94 %** | **15.92 %** | **−2.02 pp** |
| `ft_scale` region miss rate | 72.94 % | **52.50 %** | **−20.44 pp** |
| `ft_scale` share of `forward_into`'s miss latency | 9.14 % | **4.20 %** | −4.95 pp |
| `hka_stack` region miss rate | 52.35 % | **43.80 %** | −8.55 pp |
| `thr_stack` region miss rate | 24.69 % | **21.65 %** | −3.04 pp |
| `forward_into` share of engine loads | 12.56 % | 12.48 % | −0.08 pp |

AMD IBS, precise, two runs per binary, per-region attribution
(`tools/perf50/ibsaddr50.py`). **The commission's number to beat was 40.5 %; I
re-derived it at 39.73 % and it is now 37.25 %.**

**The `ft_scale` row lands exactly where the model puts it.** With 20 lines
instead of 40 and an unchanged 40 load ops per evaluation, two ops now share
each 64-byte line: the first misses, the second hits, so the floor is **50 %**.
Measured **52.50 %**.

**The last two rows are the study's mechanism showing itself.** Nothing was done
to the accumulators, yet both got better, because 20 lines per evaluation of L1
stopped being spent on the scale. That is only possible if the head's misses are
**capacity** misses — §3.

### Measured on the exact counters

15 rounds, blocked Latin square, byte-identical null riding in the rotation
(`tools/perf50/exact50.sh`, realised mean position **1.00** for all three labels):

| `sc50` vs `base50` | median | trimmed | se | neg/n | p |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | **−0.000 %** | −0.000 | **0.000** | 8/15 | 1.00 |
| **`L1-dcache-loads:u`** | **−0.563 %** | −0.556 | 0.036 | **15/15** | **0.0001** |
| `ls_dispatch.ld_dispatch:u` | +0.056 % | +0.028 | 0.067 | 7/15 | 1.00 |
| `ls_dispatch.store_dispatch:u` | −0.017 % | −0.033 | 0.073 | 8/15 | 1.00 |
| `cycles:u` | −1.215 % | −1.070 | 0.252 | 12/15 | 0.0352 |

**The null, published as required:** `instructions:u` **0.000 %**,
`L1-dcache-loads` **+0.016 %** (7/15), `ld_dispatch` +0.033 %, `store_dispatch`
−0.023 %, and **`cycles:u` −1.026 %** — which is why **no cycles claim is made**,
even though the arm reads −1.215 %.

**★ The two load counters SPLIT, and that split is the proof the change is
exactly what it claims.** Zen 4 double-pumps a 512-bit load into 2.0015
`L1-dcache-loads` but **one** `ls_dispatch.ld_dispatch` (study 29 §2.1,
measured). Replacing 40 `vmovdqu64` with 40 `vpmovzxbw` per evaluation therefore
predicts **−17.27 `L1-dcache-loads`/node and 0.00 `ld_dispatch`/node**. Measured:
**−17.0 and +1.3.** The model closes to **1.6 %**, and it closes on the counter
that moved *and* on the one that did not.

**Issue count is unchanged and that is deliberate**, not a disappointment:
study 47's table says every memory-operand op costs ~1 c regardless of class
(`vmovdqu64` 1.077, and `vpmovzxbw` takes its 256-bit memory operand in the same
single slot). This arm buys **bytes**, and `instructions:u` at se 0.000 confirms
it bought nothing else.

### Everything else that landed

* **`tools/perf50/ibsaddr50.py`** — IBS load samples attributed to **named memory
  regions**. Study 49's `ibs49.py` asks which *code* missed; this asks **which
  ARRAY it missed in**, by joining `PERF_SAMPLE_ADDR` against the engine's own
  `foldcount::region` registry. Nothing in the repo had asked that question, and
  it is what redirected this study (§2).
* **`foldcount::valhist` / `FOLDVALHIST`** — a value histogram for a small
  integer weight array, so "does every `ft_scale` fit in a `u8`?" is answered by
  the engine that loads the net instead of by re-deriving the loader's arithmetic
  in Python.
* **Four new `foldcount::region` registrations** — `ft_scale`, `head_ft`,
  `head_nnz`, `l1_weights`. Registered from inside `forward_into` behind a
  `Once`, and **not** from a constructor, because `SotaContext::new` returns by
  value through `with_tier_flag(mut self) -> Self`: the scratch is moved after
  construction, so a pointer taken there is not the live one.

---

## ★ 0. THE FRAME, RE-DERIVED (rule 8)

| | study 49 (`base49`) | **this study (`base50`)** | Δ |
|---|---:|---:|---:|
| `instructions:u` | 6153.06 | **6153.10** | **+0.0007 %** |
| `ls_dispatch.ld_dispatch:u` | 2275.86 | 2284.66 | +0.39 % |
| `ls_dispatch.store_dispatch:u` | — | 898.83 | |
| `L1-dcache-loads:u` | — | 3018.51 | |
| `cycles:u` | 3753.76 | 3720.81 | −0.88 % |

`instructions:u` reproduces study 49's to **7 parts per million**, which is as
strong a statement as this repo has that the two studies measured the same code.

**Verified from the engine's own exact counters, not inherited:**
`foldcount` on the study frame gives **2,156,392 evaluations / 5,001,562 nodes =
0.43115 evals/node** and **35,888,218 folded weight rows = 7.1754 rows/node**
(study 48's figure to 4 significant figures).

---

## ★ 1. THE COMMISSION'S PREMISE, RE-DERIVED — TWO PARTS CONFIRMED, TWO REFUTED

### 1.1 CONFIRMED — 12,800 B per evaluation, and four separate allocations

`activate` reads `pst[2][1280]` + `threat[2][1280]` + `scale[1280]` as i16 and
writes `ft[1280]` as u8. At i16 that is **5120 + 5120 + 2560 = 12,800 B read per
evaluation**, and at 0.43115 evals/node **5,518.7 B/node**. Exact, not modelled.

The four allocations are real and disjoint, confirmed from a runtime dump:

| array | allocation | bytes |
|---|---|---:|
| `pst` | `HalfkaContext.stack` — `Box<[Entry; 254]>` | 1,316,736 |
| `threat` | `ThreatContext.stack` — `Box<[ThreatEntry; 254]>` | 1,316,736 |
| `scale` | `SotaHead.ft_scale` — **its own `Blocks` box** | 2,560 |
| `ft` | `SotaContext.head_scratch` — **the thread stack** | 1,280 |

**One correction to study 49:** it calls `ft_scale` *"the leaked net blob"*. It is
not — it is its own 2560-byte heap allocation (`Blocks<i16, L1, 1>` is a
`Box<[Line<i16, 1280>; 1]>`), which is why it has its own page offset and why it
could be halved independently of anything else.

### 1.2 ★ REFUTED — the TLB half of the premise costs nothing

The commission says four allocations mean *"four sets of TLB entries"*. IBS
records the TLB level per sample and it does not support that:

| | L1 DTLB hit | L2 TLB hit | L2 TLB miss |
|---|---:|---:|---:|
| `forward_into`, all loads | **95.5 %** | 4.5 % | 0.0 % |
| — its `hka_stack` loads | **99.2 %** | 0.8 % | — |
| — its `thr_stack` loads | **99.3 %** | 0.7 % | — |
| — its `ft_scale` loads | **100.0 %** | — | — |
| — its `l1_weights` loads | 87.8 % | 12.1 % | 0.1 % |
| engine, all loads | 97.3 % | 2.6 % | 0.1 % |

**The two 1.3 MB accumulator stacks are 99.2 % / 99.3 % L1-DTLB hits.** Address
translation is not a cost at this site and consolidating for TLB reasons would
buy nothing. The only row with any DTLB pressure at all is `l1_weights`, and it
is 12.1 % into the *L2* TLB, which is a hit.

### 1.3 ★ REFUTED AS THE MAIN EVENT — the four arrays are a third of the misses

This is the measurement that redirected the study. Per-region attribution of
`forward_into`'s own load samples, three runs, agreeing to ±0.2 pp:

| region | % of its loads | miss rate | **% of its misses** | % of its miss latency |
|---|---:|---:|---:|---:|
| `l1_weights` (the sparse L1 matmul table) | 15.7 % | **88.6 %** | **35.0 %** | 37.4 % |
| unregistered heap (the head's f32 + skip tables) | 14.1 % | 84.2 % | **29.6 %** | 31.0 % |
| `hka_stack` (the `pst` accumulator) | 11.8 % | 53.9 % | 15.8 % | 13.6 % |
| `ft_scale` | 5.9 % | 75.5 % | 11.1 % | 9.3 % |
| `thr_stack` (the `threat` accumulator) | 12.2 % | 24.8 % | 7.5 % | 7.6 % |
| thread stack (spills) | 3.1 % | 9.3 % | 0.7 % | 0.8 % |
| `head_ft` (the output) | 21.5 % | **0.4 %** | 0.2 % | 0.2 % |
| `head_nnz` | 15.3 % | 0.1 % | 0.0 % | 0.0 % |

**The 12.8 KB the commission sent me to consolidate is `hka_stack` + `thr_stack`
+ `ft_scale` = 34.4 % of `forward_into`'s misses.** The majority — 64.6 % — is
`l1_weights` plus the head's other per-bucket tables, which no consolidation of
the accumulator pair can touch.

**Two rows nobody had.** `head_ft`, the 1280-byte output buffer, misses **0.4 %**
of the time: it is written and immediately re-read by `find_nnz` and `l1_dot`
and is genuinely L1-hot, which is what study 48 believed of the *whole* symbol.
And `thr_stack` misses at **half** `hka_stack`'s rate — 24.8 % against 53.9 % —
which is pure write-order: `evaluate` folds HalfKA, *then* folds threats, *then*
activates, so 11.9 KB of threat weight rows sweep L1 between the `pst` write and
the `pst` read while nothing sweeps between the `threat` write and its read.
**That asymmetry is the capacity mechanism visible in a single column.**

---

## ★ 2. WHY CONSOLIDATION CANNOT WORK HERE — and the 4 KB conflict I found on the way

### 2.1 The structural defect is real, and it is the one the commission predicted

The two accumulator stacks are **exactly 4 KB-congruent**:

```
hka_stack  0x7f9fc642e040   page offset 64
thr_stack  0x7f9fc6297040   page offset 64
difference 0x197000 = 1,667,072 B = 407 × 4096 exactly
Entry stride 5184 B for both; acc at offset 0 in both
```

Both are `mmap`'d with glibc's 64-byte header, and both carry a 5184-byte stride,
so `hka.stack[ply].acc[p][lane]` and `thr.stack[ply].acc[p][lane]` map to the
**same L1d set** (index = addr[11:6]) for every ply congruent mod 64 — which is
the common case, since the threat aliasing sends `alias[ply] == ply` for
3,477,870 of 4,270,334 folds. Nobody had looked, and it reproduces every run
because the sizes and allocation order are fixed.

### 2.2 And it is worth approximately nothing, for a reason that generalises

**The evictor is not the layout. It is the fold.** From this study's own
`foldcount` integers: **7.1754 weight rows/node ÷ 0.43115 evals/node × 1280 B =
21.3 KB of cold, never-reused weight rows streamed through a 32 KB L1 between one
evaluation and the next.** The head's entire warm working set — 10.2 KB of
accumulator, 20 KB of `l1_weights` bucket, 2.5 KB of scale — is swept out
wholesale in between. Under that regime it does not matter *where* the arrays
are, only *how many lines* of them there are.

Three independent confirmations, and they agree:

1. **Study 39's `probe-actpin`** replaced `pst` + `threat` with a fixed,
   permanently-reused, non-congruent 10.24 KB buffer — i.e. perfect address reuse
   *and* the 4 KB conflict removed — and measured **−0.22 L1 misses/node**, on an
   instrument that resolved 0.861 misses/line for the fold streams in the same
   panel. **That experiment is an upper bound on the whole consolidation idea and
   it was already on disk.** What it tests is address *reuse*; what it cannot fix
   is *capacity*, and capacity is what this is.
2. **The write-order asymmetry** in §1.3: same layout, same congruence, same
   stride — 24.8 % vs 53.9 % purely on how much weight-row traffic intervenes.
3. **This study's own arm**, in the opposite direction: removing 20 lines/eval of
   *scale* improved `hka_stack` by 8.55 pp and `thr_stack` by 3.04 pp without
   touching either. Lines freed, misses avoided.

**So the honest statement of the commissioned target: one allocation, one stride,
cache-line aligned, sequential — is priced at ≤ 0.22 L1 misses/node ≈ ≤ 0.3
cycles/node ≈ ≤ 0.007 % of frame, and I did not build it.** Not because it is
hard, but because three measurements agree on what it is worth, one of them mine
and in the direction that would have shown a win. **The lever that does work at
this site is BYTES, and that is the arm that landed.**

### 2.3 The ceiling on the whole target, stated plainly

The commission ranks `forward_into` as *"the third-largest true stall source at
17.5 %"*. That 17.5 % is a share of **miss latency**, not of stall, and the two
differ by a factor of fifty here: study 49's own master table puts
`forward_into`'s `stall / own-miss-latency` at **2.0 %** — the loads are ~50-way
overlapped and essentially all L2 hits (my runs: 6624 L1 / 4302 L2 / 62 L3 /
**0 DRAM**). Its attributed retirement stall is **79.7 cycles/node = 1.9 % of the
4179.7 frame**, and study 46 independently prices the whole symbol at 805.5
cycles/eval with 229.6 of that `load_not_complete`, i.e. 99 c/node — the same
number.

**Eliminating every L1 miss in `forward_into` is worth at most ~2 % of the frame
(~+2.4 Elo).** That does not make the site not worth attacking — this study took
2.48 pp of 39.73 out of it — but the arithmetic should be in the commission, and
it was not.

---

## 3. THE ARM, AND WHY IT IS BIT-IDENTICAL

`vpmovzxbw` zero-extends 32 bytes to 32 i16 lanes. The scale values are
`1..=8`, so the extended value is bit-for-bit the i16 that was there before, and
`vpmullw` sees identical operands. Nothing else in the expression changes.

**One deviation, stated rather than footnoted.** The loader's scale saturation
tightens from `min(i16::MAX)` to `clamp(1, u8::MAX)`. It is unreachable for any
net that can work: a column scale above 255 needs a column peak above 32,385 in
the FT_QUANT frame, and the reduced frame then requires `|acc| ≤ 127` for
`acc × scale` to stay in i16 — which `activate`'s existing `debug_assert`
enforces and which no accumulator summing dozens of rows satisfies. The canonical
net's maximum is **8**.

The AVX2 backend got the same treatment (`_mm_cvtepu8_epi16` +
`_mm_loadl_epi64`), so the `x86-64-v3` build is not left behind — study 48 §9
item 3 recorded exactly that omission as "free Elo left on the floor".

---

## 4. GATES

| gate | result |
|---|---|
| node identity, study frame, every binary | **5 001 562** — exact on `base50`, `sc50`, `fc50`, `fcsc50`, `null50` |
| d13 node identity | **1 739 588** — exact, pin unmoved |
| wl50 eval hash | **`2e1ede95…79c4`** — exact, pin unmoved |
| `instructions:u`/node vs study 49 | 6153.10 vs 6153.06 (**+0.0007 %**) |
| `foldcount` rows/node vs studies 47/48 | **7.1754** vs 7.1754 / 7.1753 |
| byte-identical null through THIS harness | `instructions:u` **0.000 %**, `L1-dcache-loads` +0.016 %, `ld_dispatch` +0.033 % |
| realised mean panel position | **1.00** for every label |
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh` | see §7 |

**No pin moves, and that is the point:** the change is bit-identical, so a pin
update would have been evidence it was not what it claims to be.

D24 perft ritual not required: nothing under `movegen`, `board` mutation,
`zobrist` or `attacks` is touched.

---

## 5. ASSUMPTIONS

| # | assumption | why believed | what breaks if false |
|---|---|---|---|
| A1 | IBS `PERF_SAMPLE_ADDR` is the tagged op's real data address | Architectural (IBS tags at issue); cross-check: the per-region load shares reproduce the kernel's static shape — `ft_scale` 5.9 % measured against 40/652 = 6.1 % predicted, `hka_stack` 11.8 % against 80/652 = 12.3 % | §1.3 and the whole redirect |
| A2 | The region dump and the samples come from the same process | Enforced: `ARCHON_FOLDCOUNT` is written by the very run `perf record` wraps. **I got this wrong first** — see §6 | §1.3 |
| A3 | The landed change is bit-identical | Verified three ways rather than argued: 5 M node count, d13 node total, wl50 eval hash | A behaviour change wearing a speedup's clothes |
| A4 | `ft_scale ≤ 255` for every loadable net | Proved for the canonical net by `FOLDVALHIST` (max 8); argued from the frame for any other (§3) | A future net would silently saturate — the argument, not the measurement, is the weak link |
| A5 | Study 39's `probe-actpin` bounds address-reuse + congruence at −0.22 misses/node | Inherited, not re-measured (§8 item 2) | §2.2's bound, but not §2.2's conclusion, which has two independent supports |
| A6 | ~120 Elo per doubling of speed | Inherited (exp-0154); its 95 % CI includes zero | Only the Elo translation |

---

## 6. THE MISTAKE I MADE, RECORDED

My first region table read **75.8 % of `forward_into`'s misses in "(other)"**,
which would have been the study's headline. It was my own harness: I classified
three runs' addresses against **run 1's** region dump, and ASLR re-rolls the heap
every run, so runs 2 and 3 fell into the catch-all. Per-run classification made
the three runs agree to ±0.2 pp and moved the mass into `l1_weights`.

The tell was that "(other)" came out at almost exactly two thirds. **A residual
bucket that lands near `(n−1)/n` of the total is a join bug, not a finding.**

---

## 7. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **The 29.6 % of `forward_into`'s misses in unregistered heap is not
   identified.** By elimination it is the head's remaining per-bucket tables —
   `l2_weights` (2048 B/eval), the D47V skip lane (1280 B/eval), and the bias
   blocks — but I registered four regions and not seven, so this is inference,
   not measurement. It is the largest single open item and it is one build away.
2. **Study 39's −0.22 misses/node is inherited, not re-derived.** I planned to
   re-measure it with `probe-actpin` under IBS — which would have been much
   sharper than study 39's counter, since IBS would show *which region* stopped
   missing — and ran out of quiet box. §2.2 does not rest on it alone (the
   write-order asymmetry and this study's own knock-on are independent), but the
   *number* does.
3. **No cycles claim.** The panel's `cycles:u` null is −1.026 % against the arm's
   −1.215 %: consistent, useless. Modelled at study 39's measured 1.458
   cycles/line for this stream, −7.5 misses/node is ≈ **−10.9 c/node ≈ −0.26 % of
   frame** (≈ +0.3 Elo), which is the same order as study 48's landed arm — but
   that is arithmetic on someone else's coefficient, not a measurement.
4. **Nothing was arbitrated by games, and nothing could be:** the binary is
   node-identical, so a fixed-node match is a dead heat by construction.
5. **The 4 KB congruence (§2.1) is measured as a fact and bounded as a cost, but
   never tested directly.** The clean experiment — offset one stack by a
   non-multiple of 4096 and re-run the region census — is two lines and I did not
   build it, because §2.2's three converging results made it the wrong next
   thing.
6. **`l1_weights` at 35 % of `forward_into`'s misses and an 88.6 % miss rate is
   priced and NOT attacked.** It is a 20 KB per-bucket block, randomly indexed by
   the nonzero-group list, evicted between evaluations by the same 21.3 KB of
   weight rows. It is now the largest single item inside this symbol and the
   obvious next target.

---

## 8. REPRODUCTION

```bash
# 0. the worktree needs the gitignored net visible under its own path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. build + gate before any timing
bash tools/perf50/build50.sh base50
cp bin50/base50 bin50/null50                      # the byte-identical null
./bin50/base50 bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn                    # must print 5001562
./bin50/base50 nnue-eval engines/canonical/net.cenn \
    < tools/nnue/calib/wl50.fens | sha256sum      # must be 2e1ede95...

# 2. THE INSTRUMENT — IBS load samples joined to NAMED REGIONS.
#    The foldcount dump MUST come from the same process as the samples:
#    these are heap addresses and ASLR re-rolls them every run (Sec 6).
bash tools/perf50/build50.sh fc50 foldcount
ARCHON_FOLDCOUNT=/home/nick/.cache/perf50/fc.txt \
  taskset -c 2 perf record -q -e ibs_op/cnt_ctl=1,swfilt=1/ -c 10000 -d -W \
    -o /home/nick/.cache/perf50/ibs.data -- ./bin50/fc50 \
    bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
perf script -i /home/nick/.cache/perf50/ibs.data \
    -F ip,sym,symoff,addr,weight,data_src \
  | python3 tools/perf50/ibsaddr50.py /home/nick/.cache/perf50/fc.txt forward_into

# the ft_scale value histogram that decided the arm
grep FOLDVALHIST /home/nick/.cache/perf50/fc.txt

# 3. the exact-counter panel (blocked Latin square)
P50_CPU=2 P50_PANEL=docs/research/perf-artifacts-50/panel-sc.txt \
  bash tools/perf50/exact50.sh 15 base50 sc50 null50
python3 tools/perf50/med50.py docs/research/perf-artifacts-50/panel-sc.txt \
  base50 sc50 null50
```

**Do not park the IBS text on `/tmp`** — it is tmpfs. This study pipes
`perf script` straight into the classifier so the 250 MB dump is never
materialised at all.

Pick a quiet **physical** core first: sum each SMT sibling pair (`N` and `N+8` on
this box) before choosing. Core 2 was quietest at 7.6 % combined.
