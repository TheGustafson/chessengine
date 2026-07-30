# Perf study 41 — the front end, measured

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 16 GB ·
**Base:** `main` at `56fc4c3` (study 40 merged) · **Net/arch:** D48X @
`network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf41/` · **Artifacts:** `docs/research/perf-artifacts-41/`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO, pinned to
core 2 with `taskset`. **Every binary here is node-identical at 5,001,562 and
eval-hash-identical at `2e1ede9521624dd8`**, with `instructions:u` matching to
**0.019 % or better** — gated per binary, not assumed
(`tools/perf41/identalign41.sh`).

**The reference reproduces study 40 on the two levels this study is built on**,
which is the frame check that matters most:

| per node | study 40 (n = 159) | this study (n = 110) |
|---|---:|---:|
| `ex_ret_brn_misp:u` | 24.092 | **24.336** (+1.0 %) |
| `ex_no_retire.empty:u` | 192.907 | **205.333** (+6.4 %, busier box) |

### 0.1 The denominator, stated — and which binary it belongs to

Every percentage below divides by a number measured **on this study's own
binary, in this study's own panel**, never inherited:

| `bin34/b41` — `main` @ `56fc4c3`, non-PGO, `-C target-cpu=native` | per node |
|---|---:|
| `cycles:u` | **4304.53** |
| `instructions:u` | **5928.40** |
| IPC | **1.377** |
| `ex_ret_brn_misp:u` | **24.336** |
| `ex_no_retire.empty:u` | **205.333** |

Source: `docs/research/perf-artifacts-41/dose3.txt`, median over **110 paired
rounds**, box carrying two perf studies and the D49 trainer. `cycles:u` can only
be inflated by contention, so 4304.53 is an upper bound on the quiet-box figure
and study 40's implied ~4104 is the lower one; **every cycle percentage in this
document is therefore a slight UNDER-statement**, which is the safe direction
for a study whose job is to bound a prize.

**Perf study 36's 8262.9 cycles/node is not used anywhere in this study** — no
result here was derived from it and none needs re-deriving
(`grep -rn 8262 tools/perf41 docs/research/perf-study-41-*` is empty).

**A binary-identity note, because the coordinator's independent measurement of
the same frame reads differently and both numbers are right.** Measured on
`engines/canonical/archon` (the shipped PGO binary, `engines/bin/archon-s25`,
built 2026-07-29 22:53, `.text` 699 893 B): 4489.8 cycles, 5659.0 instructions,
22.35 mispredicts per node over **5,001,566** nodes. That is a *different
binary*, and the node count proves it — PGO cannot change a fixed-node search's
node count, so 5,001,566 ≠ 5,001,562 means different source, not different
optimisation. Against `b41` it retires 4.5 % fewer instructions and mispredicts
8.2 % less.

It does not move this study's conclusion, and the reason is worth stating:
study 40's 9.13 cycles-per-mispredict was measured on the **same non-PGO frame
as `b41`**, so `24.336 × 9.13` is like-for-like. Run the same arithmetic on the
shipped binary instead — `22.35 × 9.13 = 204.0` — and mispredicts still account
for essentially the whole `ex_no_retire.empty` bucket. **The headline survives
either binary.**

---

## ★ THE DECOMPOSITION

**`ex_no_retire.empty` is instruction-FETCH-bound, and mispredicts already
over-explain it.** The commission's premise — that mispredicts account for only
a small fraction of the 192.91 — is **falsified by study 40's own two numbers
multiplied together**, and this study reproduces both of them:

> 24.336 mispredicts/node × **9.13 cycles each** (study 40 §3.3, measured, and
> flagged there as a *lower* bound) = **222.2 cycles/node**, against an
> `ex_no_retire.empty` of **205.333**. That is **108 % of the bucket** before
> anything else is charged to it.

| channel | measured on | share of `ex_no_retire.empty` | verdict |
|---|---|---:|---|
| **branch mispredicts** | 24.336/node × 9.13 c (study 40's arm, a different family) | **≥ 108 %** | the bucket |
| **instruction fetch pressure** | dose response, §3 — 3 gated layout arms, 19–37 se on the fetch counters | **~ 92 % local elasticity** | the *same* currency |
| **L1i misses specifically** | β = 0.450 ± 0.081 c/miss at the margin (§3.2) | **≤ 37 %**, and NOT extrapolable — §3.3 | over-priced by linearity |
| **instruction TLB** | 0.787 L2-ITLB hits + 0.0063 walks per node, direct | **≤ 3.2 %** | **DEAD — do not build the hugepage arm** |
| **decode bandwidth / loop buffer** | loop buffer delivers **exactly 0** ops/node | **~ 0** | wrong bucket by construction |
| **unattributed residual** | — | **none left; the terms OVERLAP by ~50 %** | §2.4 |

**The residual is negative, and that is the finding.** Mispredicts and fetch
pressure are not two causes to be split — they are one. A mispredict's cost is
a refetch, and a refetch on this engine walks into an L1 instruction cache the
hot set overflows **2.0×** (§4). Charge the marginal rate of either channel over
the whole bucket and you get ~100 % twice, because the same cycles carry both
labels.

### What that closes

* **Hugepages for `.text`: dead, measured.** 1430.3 L1-ITLB fetch hits per node
  against 0.787 misses — a **99.945 %** hit rate. Even pricing every miss at a
  full L2-ITLB walk, address translation cannot be more than 3.2 % of the
  bucket. No arm needed; the counter settles it.
* **LLVM hot-cold splitting: dead, measured, and it goes the WRONG WAY.**
  `hcs41` (`-hot-cold-split -enable-cold-section`) makes instruction cache
  misses **worse** by +1.485 % ± 0.229 % (**6.5 se**, n = 110) for a 2.6 %
  `.text` reduction, and does not move `ex_no_retire.empty` (+0.961 % ± 0.506 %,
  1.9 se).
* **Outlining cold code out of hot functions: no prize, measured.** The hot set
  allocates 82.8 KB but **executes 65.2 KB of it** (§4) — the other 21 % is
  already never fetched, so moving it cannot reduce a fetch. `search::<NonPv>`
  is 83.9 % executed.
* **Code layout in general: the dose SATURATES** (§3.3), so no layout arm's
  benefit can be extrapolated from a degradation, and every layout degradation
  measured here costs less per unit than the linear model predicts.

### The one number a future study should take

**`ex_no_retire.empty` moves about one-for-one, in percent, with
`ic_tag_hit_miss.all_instruction_cache_accesses`** — the count of 64-byte
instruction fetches, whatever causes them. Measured at the smallest dose
(+3.000 % ± 0.221 % accesses → +2.770 % ± 0.885 % empty, ratio **0.92**), and
the ratio *falls* as the dose grows, so 0.92 is a floor for the local slope.

That is the exchange rate for the whole front end, and it says the lever is
**fewer instruction fetches** — by mispredicting less, or by executing fewer
code bytes. It is emphatically *not* by moving code around.

---

*(sections 1–7 below)*
