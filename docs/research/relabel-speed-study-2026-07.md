# SF18 Relabel Speed Study — beyond warm-TT (2026-07-19)

**Question:** how fast can the ~230M-position SF18 canonical relabel
(`docs/research/sf18-canonical-epic.md` Phase A) go, beyond the already-adopted
warm-TT trick, at acceptable label quality?

**Method:** every lever measured empirically on the local 7700X (Zen 4, WSL2),
same harness as the original warm study — `root-analysis --reference-only`
over game-ordered UHO FENs, labels compared against a cold@50k reference.
All numbers in this doc were produced this session; the harness, raw runs, and
exact commands are in the appendix.

---

## 1. Executive summary

**The two levers that matter are free: drop the budget from warm@20k to
warm@10k (labels statistically identical), and rebuild SF18 `x86-64-bmi2` +
PGO for the c2d fleet (labels bit-identical, +6.7% NPS).** Together they take
the full relabel from the planned ~18 h / ~$20 to **~9.5 h / ~$10.5** — a
**1.9× improvement over the current plan, 5.35× over cold@50k** — with *zero*
measurable label-quality cost relative to the warm@20k plan. Syzygy-routing
the 18.6% endgame mass adds another ~1.2× if we accept a label-semantics
decision. Everything else measured (bigger hash, threads, incremental feed,
UCI options, cheap-screen cascade) is a measured **no** — the numbers are below.

### Ranked lever table (all measured, n=988 game-ordered UHO positions)

| # | Lever | Speedup (measured) | Label cost vs cold@50k | Effort | Verdict |
|---|-------|-------------------|------------------------|--------|---------|
| 1 | **Node budget: warm@20k → warm@10k** | **1.81×** (12.94→7.22 s/1000) | dq 0.0101→**0.0097** (better/equal); median Δ 12→11cp; gated flips 0→1 per 988 | config only | **DO NOW** |
| 2 | **SF18 build: avx2 → bmi2+PGO** (c2d ceiling) | **1.067×** NPS + wall confirmed (7.22→6.83 s) | **zero — labels bit-identical** (r=1.0000, Δ=0 on all 988) | one `make` + image re-pin (image rebuild already required for `--warm`) | **DO NOW** |
| 3 | **Syzygy-route ≤7-man positions** (18.6% of corpus × 0.87 rel. cost) | **1.19×** (skip 16% of wall) | exact-WDL labels for routed rows — a semantics *decision*, not a degradation | TB files + manifest split + label pass | LATER (worth it) |
| 4 | **Hardware: c3d (Genoa, AVX-512/VNNI) probe** | ≥1.06× per ISA alone (vnni512-pgo +13.2% vs avx2, +6% vs bmi2-pgo) + unmeasured Genoa IPC/clock; spot $/vCPU ≈ parity | zero (labels arch-independent, proven) | 1 h spot probe ≈ $0.20 | LATER (cheap probe) |
| 5 | Manifest dedup (board+stm) | ~1.01× (1.12% dup in 2M-row sample; global likely a bit more) | zero | trivial at manifest build | do with manifest |
| 6 | Aggressive budget: warm@5k | 3.41× vs warm@20k (3.79 s/1000) | dq 0.0114 (+18% vs warm@10k; still 32% better than the binpack-native labels the winning nets trained on) | config only | fallback / Phase-E option |
| 7 | Hash > 16MB | **0.96×** (slower: 7.22→7.52→7.78 s for 16/64/256MB) | no gain (dq 0.0097 / 0.0103 / 0.0107) | — | **REJECT — 16MB is optimal** |
| 8 | Threads > 1 per position | T4 = 2.88× on 4 cores = **0.72 per-core efficiency** | worse (dq 0.0122 @T4) + nondeterministic | — | **REJECT — 1×N wins** |
| 9 | Cheap-screen cascade (2k→10k escalation) | 1.51× at best deployable rule | dq 0.0118 — **dominated by flat warm@5k** (faster AND better) | driver change | **REJECT v1** |
| 10 | Incremental move-feed / UCI pipeline | <1.05× ceiling (overhead floor = 0.61 ms/pos, 8% of warm@10k wall, incl. shard write) | — | driver + desync risk | **REJECT** |
| 11 | UCI eval/speed options | none exist (full option scan below) | — | — | n/a |

### Recommended stacked config (Phase A)

> **SF18 rebuilt `make profile-build ARCH=x86-64-bmi2` · warm@10k · Hash=16 ·
> Threads=1 · 1 proc/vCPU · game-ordered manifest, deduped**
>
> = **5.35× over cold@50k, 1.9× over the current warm@20k plan**
> = **~9.5 h, ~$10.5** for the 230M relabel at the 96-vCPU cap
> Labels: median |Δ| 11cp vs cold@50k, mean sigmoid-space |Δq| 0.0097,
> 1 winner-flip per 988 at |ref|≥50cp — statistically indistinguishable from
> the validated warm@20k plan (0.0101 / 12cp / 0 flips).
>
> With the Syzygy split (later): **~8.0 h / ~$8.9** → ~2.3× over plan.
> If the c3d probe pays off: multiply a further ~0.9×.

---

## 2. Method

- **Positions:** first 1000 game-ordered FENs of `nodes5000pv2_UHO.binpack`
  via `fast_fens` (10 games, consecutive plies — the actual Phase-A access
  pattern). The original 200-FEN set reproduced the epic's warm@20k numbers
  (median Δ14cp, 0 flips) before scaling up.
- **Driver:** `target/release/root-analysis --reference-only` (the production
  relabeler), `--uci-option Threads=1 --uci-option Hash=16` unless swept.
- **Timing:** single pinned core (`taskset -c 15 nice -n 19`), idle box for
  all `i-*` runs (the SPRT that was running finished before the final matrix;
  headline runs repeated: warm@10k = 7.22/7.13/7.21 s, cold@50k = 36.56/36.44 s
  → ±0.6% noise).
- **Quality metrics vs cold@50k:** median/mean |Δcp|; strict sign flips and
  gated flips (|ref|≥50cp); Pearson r; and **mean |Δq| where q=σ(cp/400)** —
  the sigmoid-space perturbation, i.e. what the trainer actually sees under
  the canonical scale=400 target. dq is the decision metric: raw-cp outliers
  live almost entirely in already-decided positions (see §4.2).
- **Reference validity:** cold@50k re-run is bit-identical (Δ=0 on 988), so
  the reference itself is deterministic.

---

## 3. Lever 1 — SF18 build flags

### 3.1 What the fleet runs today

`engines/stockfish-18-src/src/stockfish` — the binary pinned into the worker
image (`tools/cloud/worker/image_manifest.json` → `bin/stockfish-18`) — is
compiled **`x86-64-avx2`**. The c2d fleet (EPYC Milan, Zen 3) supports
BMI2/fast-PEXT; the local 7700X additionally has full AVX-512 + VNNI.

### 3.2 Measured NPS by build (SF `bench`, median of 5–9 interleaved runs)

Every build produced the **identical bench node count (2,050,811)** — SF's
search at fixed nodes is bit-exact across SIMD paths, so builds change speed
only, never labels. Confirmed end-to-end: warm@10k labels under
`vnni512-pgo` vs `avx2` → r=1.0000, every Δcp = 0.

| Build | NPS (median) | vs avx2 | Notes |
|---|---|---|---|
| x86-64-avx2 (current) | 1,443k | — | what the fleet runs now |
| x86-64-avx2 + PGO | 1,516k | +5.0% | PGO alone |
| x86-64-bmi2 | 1,460k | +1.1% | PEXT movegen (fast on Zen 3+) |
| **x86-64-bmi2 + PGO** | **1,541k** | **+6.7%** | **= c2d/Milan ceiling** |
| x86-64-avx512 | 1,516k | +5.0% | 512-bit NNUE |
| x86-64-vnni512 | 1,540k | +6.6% | + VNNI dpbusd |
| **x86-64-vnni512 + PGO** | **1,634k** | **+13.2%** | Zen 4 / c3d ceiling |

Decomposition: PGO ≈ +5% on every arch; ISA gives +1% (bmi2), +5% (avx512),
+6.6% (vnni512). The "AVX-512 will be huge" hypothesis is **wrong for SF18** —
its AVX2 NNUE path is already near-optimal and relabel searches are TT-heavy;
the honest ISA win on hardware that has it is ~6–8% over bmi2.

End-to-end confirmation (warm@10k, 1000 FENs): avx2 7.22 s → bmi2-pgo
**6.83 s** (+5.7%) → vnni512-pgo **6.39 s** (+13%). Bench deltas transfer ~1:1.

### 3.3 What to build for the c2d fleet

**`make -j profile-build ARCH=x86-64-bmi2 COMP=gcc`.** Milan has no AVX-512,
so bmi2+PGO is the ceiling there. The PGO profile was generated on the local
Zen 4 — profile data is input-driven (bench positions), not arch-specific;
this is standard practice. Verification on the worker is one line
(`echo compiler | stockfish` must print `x86-64-bmi2`), and the
labels-bit-identical proof means Gate A's "ratio 1.000 vs live SF18" check is
unaffected. The worker image must be rebuilt for `--warm` anyway — re-pinning
the SF sha in `image_manifest.json` to the bmi2-pgo binary is **zero marginal
ops**.

If the fleet ever runs Genoa/Turin (c3d/c4d, see §10): pin `vnni512-pgo`
instead (runtime-gated per machine family — the image can carry both binaries
and pick by `lscpu` flag `avx512_vnni`).

---

## 4. Lever 2 — node budget (the big one)

### 4.1 The measured budget/quality frontier (n=988, ref = cold@50k)

| Config | s/1000 pos | × vs cold50k | med Δcp | mean Δcp | p90 | flips (any / ≥50cp) | r | **mean dq** |
|---|---|---|---|---|---|---|---|---|
| cold@50k | 36.5 | 1.00 | — | — | — | — | — | — |
| cold@20k | 15.78 | 2.31 | 9 | 23.9 | 42 | 25 / 0 | .947 | 0.0085 |
| **warm@20k** (plan) | 12.94 | 2.82 | 12 | 26.0 | 46 | 27 / 0 | .946 | 0.0101 |
| cold@10k | 9.66 | 3.78 | 14 | 36.0 | 62 | 39 / 0 | .906 | 0.0133 |
| **warm@10k** ★ | **7.22** | **5.06** | **11** | 24.9 | 41 | 17 / 1 | .938 | **0.0097** |
| cold@5k (binpack-native class) | 6.02 | 6.07 | 16 | 43.1 | 81 | 48 / 1 | .883 | 0.0168 |
| warm@5k | 3.79 | 9.64 | 12 | 32.4 | 47 | 30 / 1 | .884 | 0.0114 |
| warm@2k | 1.81 | 20.2 | 14 | 42.3 | 74 | 57 / 2 | .870 | 0.0166 |
| warm@1k | 1.17 | 31.2 | 18 | 51.1 | 105 | 57 / 2 | .858 | 0.0211 |

Key facts:

1. **warm@10k ≡ warm@20k in label quality** (dq 0.0097 vs 0.0101, median 11
   vs 12cp — inside noise) at **1.81× less cost**. The extra 10k nodes buy
   nothing measurable. warm@20k was a safe first pick; the curve says 10k is
   the right knee.
2. **Warm beats cold decisively at equal nodes** — warm@5k dq 0.0114 vs
   cold@5k 0.0168, same node count. TT inheritance is a real effective-depth
   boost, not just a wall-clock trick.
3. **The cold-yardstick slightly penalizes warm** — cold@20k (0.0085) scores
   "better" than warm@20k (0.0101) against a *cold* 50k reference, because
   warm labels are history-dependent by design (accepted 2026-07-19). The
   ~0.0015 dq offset is bias relative to cold semantics, not noise.
4. **Context anchor:** the nets that beat net-0 trained on binpack-native
   labels ≈ cold@5k class, dq **0.0168**. Everything at warm@5k and above is
   *substantially cleaner than the labels that already won*. warm@2k merely
   ties the binpack class — at 20× the throughput of cold@50k.
5. Flip anatomy: raw sign flips (1.7–5.8%) are almost all in the |ref|<50cp
   dead zone; gated flips stay ≤2/988 down to warm@2k. Winner identity is
   robust at every budget ≥2k.

### 4.2 Where the residual error lives

The big |Δcp| outliers (>300cp: 4–5 per 988 at 10–20k) sit in positions whose
reference eval is **already ±1700–2400cp** — magnitude disagreement in decided
games. In sigmoid space they are invisible (hence dq as decision metric).
By game phase (warm@10k): mean dq = 0.0111 (21+ men) / 0.0088 (13–20) /
0.0096 (8–12) / **0.0059 (≤7 men)**. Error is *flat-to-decreasing* toward the
endgame — there is **no case for phase-shaped budgets**, and Syzygy's value
(§7) is compute-skipping, not accuracy repair.

**Recommendation: warm@10k.** Fallback if Phase-A time ever matters more than
the last bit of label polish: warm@5k at dq 0.0114 is a defensible 9.6×.

---

## 5. Lever 3 — Hash size: measured REJECT

warm@10k, 1000 FENs: Hash 16MB → 7.22 s, dq 0.0097 · 64MB → 7.52 s, 0.0103 ·
256MB → 7.78 s, 0.0107. warm@20k with 256MB: 14.25 s vs 12.94 s, dq 0.0104 vs
0.0101. Bigger hash is **slower** (TT walks lose cache locality; a 10k-node
search touches ~1MB of entries) and buys **zero quality** — 16MB already
retains several plies of game history for the warm chain. Keep **Hash=16**.
(Also the fleet-friendly answer: 16 procs × 16MB = 256MB on a 32GB VM.)

## 6. Lever 4 — Threads: measured REJECT

warm@10k per position, total nodes fixed: T1 = 7.22 s · T2 = 4.49 s (1.61×,
0.80/core) · T4 = 2.51 s (2.88×, **0.72/core**) — and quality *drops* (dq
0.0111 @T2, 0.0122 @T4; SMP widening + nondeterminism). 16 × 1-thread procs
out-throughput 4 × 4-thread procs by ~39%. **1 thread/proc × 1 proc/vCPU
stands.**

## 7. Lever 5 — position selection

### 7.1 Syzygy routing (real, medium effort)

Measured endgame mass in the actual Phase-A corpus (500k–2M row samples):

| Corpus | ≤5 men | ≤6 men | ≤7 men |
|---|---|---|---|
| UHO (168.5M after c1 cut) | 6.5% | 11.9% | 18.1% |
| wrongIsRight (32M) | — | 19.2% | 26.9% |
| farseer (30M) | — | 9.5% | 12.3% |
| **weighted 230M** | ~6% | **~12.6%** | **~18.6%** |

Per-position cost at fixed nodes is 0.87× for ≤7-man vs mixed (9.0 vs 10.4
ms/pos measured), so routing ≤7-man away saves ≈ 18.6% × 0.87 ≈ **16% of
fleet wall-clock (×1.19)**; ≤6-man only ≈ ×1.12. The routed rows get
**tablebase-exact WDL** — a strictly better verdict than any SF search, but a
**label-semantics decision**: cp must be synthesized (e.g. WDL-saturated ±S,
or TB-WDL as the training target directly with the cp term masked). That
decision belongs with the Phase-B target design, so this lever is **later,
not now**. Infra: 6-man WDL ≈ 68GB (fits neither the local disk today at 98%
nor comfortably in the worker image; a GCS-mounted read-only disk on one
dedicated VM, or a local pass after pruning, both work — the probe itself is
µs/position, so one machine covers 43M rows in hours).

### 7.2 Dedup (small, free)

1.12% exact (board+stm) duplicates within a 2M-row consecutive sample;
`obs-tool export-value` already drops in-shard dups (12/1000 in this study's
runs) but the fleet still *searches* them. Dedup at manifest build is free —
do it there (keep game order; dropping repeats does not break the warm chain).
Global cross-corpus dup rate will be somewhat higher (shared opening lines);
even 3% is $0.3 — take it, but it's not a headline.

### 7.3 Cheap-screen cascade: measured REJECT (v1)

Measured with the production driver's own multi-tier mode (`--nodes
2000,10000 --warm` — same process, shared TT = exactly the deployable cost
structure): the 2k pre-pass adds only **1.16 s/1000 (16%)** ahead of a 10k
search, and marginal-10k-after-2k = 6.57 s/1000. Escalation-rule simulation
against cold@50k truth:

| Rule (escalate to 10k when…) | escalated f | s/1000 | × vs flat 10k | dq |
|---|---|---|---|---|
| always (= flat 10k) | 1.00 | 7.22 | 1.00 | 0.0097 |
| \|cp₂ₖ\| < 400 | 0.80 | 7.08 | 1.02 | 0.0109 |
| \|cp₂ₖ\| < 200 | 0.68 | 6.28 | 1.15 | 0.0114 |
| \|cp₂ₖ\| < 100 | 0.45 | 4.78 | 1.51 | 0.0118 |
| oracle top-30% true error | 0.30 | 3.78 | 1.91 | 0.0095 |
| **flat warm@5k (no cascade)** | — | **3.79** | **1.90** | **0.0114** |

The best deployable magnitude rule (1.51×, dq 0.0118) is **strictly dominated
by just running flat warm@5k** (1.90×, dq 0.0114). The warm chain already
smooths quality so well that flat budgets sit on the frontier; only an
oracle-grade instability signal would beat them (the id-trajectory features in
the Tier-F shards are the candidate — a *later* research idea, not a Phase-A
lever).

## 8. Lever 6+7 — incremental feed & pipeline overhead: measured REJECT

Overhead floor via `--nodes 1`: warm = **0.61 ms/pos** — that is the entire
non-search pipeline (UCI round-trip, `position fen` parse, info parsing, obs
episode build + shard write) = 8% of warm@10k wall on the 7700X (~4–5% on
slower c2d cores). Cold mode adds +1.77 ms/pos for `ucinewgame`+`isready` —
warm already deleted that. A make-move feed (`position fen X moves m`) could
only shave a fraction of the 0.61 ms (SF rebuilds accumulators from root per
`position` regardless) → **<5% ceiling for real driver work + a position
desync risk class**. Not worth it. Engine-process reuse across a whole shard
is already how the driver works; shard I/O is 561 B/pos (~129GB total at
230M — a GCS storage note, not a throughput factor).

## 9. Lever 9 — UCI/eval options: nothing there

Full SF18 option scan (`uci`): `Threads/Hash/MultiPV/Ponder/Move
Overhead/nodestime/Skill Level/UCI_LimitStrength/UCI_Elo/UCI_ShowWDL/
NumaPolicy/SyzygyPath*/EvalFile/EvalFileSmall/Debug Log File`. MultiPV=1 and
ShowWDL=off are already defaults; `Skill Level`/`UCI_Elo` inject *randomized*
weakening (never acceptable for labels); `nodestime` is a TC shim; SF18 has no
Contempt; the small-net-for-decided-positions trick is automatic
(`EvalFileSmall`) and already part of every number in this study. `NumaPolicy`
is a no-op on single-socket VMs. **No accuracy-for-speed dial exists beyond
nodes/Hash/Threads/Syzygy — all covered above.**

## 10. Lever 8 — hardware shape

- Current: c2d-highcpu-{8,16,32} spot (EPYC Milan/Zen 3), per-vCPU-flat
  pricing already exploited by the one-task-per-vCPU packing
  (`submit_obs_relabel.py`); pinned price $0.183008/h per 16 vCPU =
  **$0.01144/vCPU-h**; the 96-vCPU `CPUS_ALL_REGIONS` cap is global and
  permanent, so *throughput per vCPU is the only free axis* — which is exactly
  what levers 1–3 buy.
- **c3d (Genoa, Zen 4)** supports AVX-512+VNNI → vnni512-pgo (+6% over
  bmi2-pgo from ISA alone, measured on Zen 4) *plus* an unmeasured Genoa vs
  Milan IPC/clock gain (typically double-digit %). Public spot trackers put
  c3d-highcpu within a few % of c2d per vCPU (region/day-volatile; both
  ~$0.0075–0.0115/vCPU-h band). **Action: a 1-hour c3d-highcpu-16 spot probe
  (~$0.20) running the vnni512-pgo binary on a real shard → rows/s/vCPU and
  live spot price → switch iff $/Mpos wins.** c4d (Turin/Zen 5) same probe
  logic. Quota counts vCPUs identically, so a per-vCPU throughput win converts
  1:1 into wall-clock at the cap.
- Keep highcpu (2GB/vCPU is 100× the 16MB/proc need); no case for other
  families at flat-or-worse per-vCPU spot pricing.

Pricing sources: [GCP spot VM pricing](https://cloud.google.com/spot-vms/pricing),
[compute-optimized pricing](https://cloud.google.com/products/compute/pricing/compute-optimized),
[cloudprice.net c3d](https://cloudprice.net/gcp/compute/instances/c3d-highcpu-4),
[holori c3d-highcpu-16](https://calculator.holori.com/gcp/vm/c3d-highcpu-16?region=us-central1).

---

## 11. Projections for the 230M relabel (96 vCPU)

Anchored to the epic's own warm@20k estimate (18 h / $20 ⇒ 27 ms/pos on c2d =
2.09× the local 7700X ms/pos; local ratios transfer because the workload is
NPS-bound — verified by builds transferring bench-delta → wall-delta 1:1).

| Stack | local ms/pos | fleet hours | fleet $ | × vs plan | label quality (dq / med Δcp / gated flips) |
|---|---|---|---|---|---|
| warm@20k · avx2 (current plan) | 12.94 | ~18 | ~$20 | 1.0× | 0.0101 / 12 / 0 |
| warm@10k · avx2 | 7.22 | 10.0 | $11.1 | 1.8× | 0.0097 / 11 / 1 |
| **warm@10k · bmi2-pgo** ★ | **6.83** | **9.5** | **$10.5** | **1.9×** | **identical to row above** |
| + ≤7-man Syzygy route | 5.72 eff. | 8.0 | $8.9 | 2.3× | + exact TB labels on 18.6% |
| warm@5k · bmi2-pgo (aggressive) | 3.59 | 5.0 | $5.5 | 3.6× | 0.0114 / 12 / 1 |
| c3d probe upside (vnni512-pgo, ISA only) | 6.39 | ~8.9 | ~$9.8* | ~2.0× | identical |

\* at c2d-parity spot pricing; the probe measures the real number (Genoa IPC
should improve it further).

---

## 12. Implement now vs later

**Now (both zero-risk, both ride the already-planned image rebuild):**
1. `make -j profile-build ARCH=x86-64-bmi2 COMP=gcc` in
   `engines/stockfish-18-src/src/`, re-pin its sha256 in
   `tools/cloud/worker/image_manifest.json`, rebuild the worker image (already
   required for `--warm`), verify `echo compiler | stockfish` on a worker.
2. Set the Phase-A campaign to `--nodes 10000 --warm` (Hash=16, Threads=1
   unchanged). Update the epic's warm@20k references to warm@10k with this
   study as evidence.
3. Dedup (board+stm) when building the FEN manifest; keep game order.

**Later / gated:**
4. c3d-highcpu-16 spot probe with the vnni512-pgo build (1 h, ~$0.20);
   switch machine list if $/Mpos wins. Same for c4d when visible in-region.
5. Syzygy ≤6/7-man routing — decide TB-label semantics in Phase-B target
   design first; then it's a manifest split + a µs/position probe pass
   (~1.2× fleet win).
6. Trajectory-feature escalation (the only cascade that could beat flat
   budgets — the id_snapshots are already in every Tier-F shard); research
   item, not Phase A.

**Measured rejects (do not revisit without new evidence):** Hash >16MB,
Threads >1, magnitude-rule cascades, incremental move-feed, UCI option
tweaks, phase-shaped budgets.

---

## Appendix A — raw measurements

### A.1 Environment
- Local: AMD Ryzen 7 7700X (Zen 4, AVX-512+VNNI), WSL2, 16 threads; runs
  pinned `taskset -c 15 nice -n 19`; headline matrix on an idle box
  (repeats: warm@10k 7.22/7.13/7.21 s; cold@50k 36.56/36.44 s).
- Driver: `root-analysis` (built 2026-07-19, `--warm` + `--reference-only`),
  labels via `obs-tool export-value`.
- SF18 source builds: gcc 13.3.0, `engines/stockfish-18-src/`; every variant
  verified via its `compiler` banner; original pinned avx2 binary
  (sha256 3a4292…99d0) restored after the build matrix — verified identical.

### A.2 Determinism proofs
- cold@50k repeat: Δ=0 on 988/988.
- warm@10k repeat: Δ=0 on 988/988 (the warm chain is deterministic given
  order + binary).
- warm@10k avx2 vs vnni512-pgo: Δ=0 on 988/988 (labels are build-independent).
- All 7 builds: SF bench nodes = 2,050,811 exactly.

### A.3 Reproduction commands
```bash
# FEN set (game-ordered)
tools/nnue/sfbinpack-audit/target/release/fast_fens \
  data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack \
  200000000 | head -1000 | cut -f1 > game1000.fens

# One measured run (sweep nodes/hash/threads/engine as per tables)
taskset -c 15 nice -n 19 target/release/root-analysis \
  --engine <stockfish-variant> --fens game1000.fens --nodes 10000 --warm \
  --reference-only --uci-option Threads=1 --uci-option Hash=16 --out run.ceobs
target/release/obs-tool export-value run.txt run.ceobs

# Build variants (in engines/stockfish-18-src/src)
make -j profile-build ARCH=x86-64-bmi2 COMP=gcc     # fleet (c2d/Milan)
make -j profile-build ARCH=x86-64-vnni512 COMP=gcc  # Zen4 / c3d probe

# Quality metric: join ref/test on FEN, report median|dcp|, sign flips,
# gated flips (|ref|>=50), Pearson r, and mean |sigma(cp/400)| delta (dq).
# Cascade cost structure: --nodes 2000,10000 --warm (same process, shared TT).
```

### A.4 Corpus statistics (samples)
- UHO 2M rows: ≤5-man 6.47%, ≤6 11.90%, ≤7 18.13%, ≤10 35.47%;
  dup(board+stm) 1.12%.
- farseerT76 500k: ≤6 9.47%, ≤7 12.32%. wrongIsRight 500k: ≤6 19.20%, ≤7 26.86%.
- Endgame (≤7-man) per-position cost at fixed nodes: 0.87× of mixed.
- Tier-F shard size: 561 B/pos (≈129GB for 230M, GCS).
