# Perf study 63 — the arbiter had two defects; fixed, it kills the three rungs

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`16098195`**
(post-study-62 handoff) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Artifacts:** `docs/research/perf-artifacts-63/`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT,
`-C target-cpu=native`. Unlike studies 60–62 **the measured pair is PGO** — the
configuration the engine ships in and the one the studies 61+62 handoff priced
at +3.14 Elo. A GPU training run (sb ~1165 of 1700) shared the box throughout
and was the only other load; every panel carries a `uptime` + top-CPU census
immediately before and after it in `perf-artifacts-63/box-quietness.txt`, and
the process census is identical across all of them.

---

## ★ PRE-REGISTRATION — committed at `a6959dc5` before any wall-clock number

Study 62 dropped three rungs on a **proxy** — `>= 2.0 i/node per KB of .text` —
and refused to relax it after its own front-end panel argued the proxy was too
strict. That refusal was right, and nothing here re-argues the threshold. This
study measures the quantity the threshold stood in for.

**The pair.** `p63-base` (today's `main`) against `p63-all`
(`shape62e` + `dual62b` = all three dropped rungs), both native + PGO with the
canonical net.

**The instrument.** `tools/wallclock_panel.py`, pre-registered median-only
licence unchanged, byte-identical null interleaved inside every round.

**The round count, and why it is not 40.** Sign-test power scales as the SQUARE
of the contrast. The tool's 40-round default is calibrated in its own docstring
for the ~1 % arms of studies 52–58; this arm is **−0.6513 % of PGO
instructions**, which at the handoff's 2.08× instructions-overstate-time ratio
predicts **≈ −0.313 % of wall clock** — so ~9× the rounds. Pre-registered:
**400 rounds** full frame and **400** at `--startup-frame`.

**The decision rule.** (1) NOT LICENSED ⇒ quote nothing, land nothing.
(2) LICENSED ⇒ subtract startup **in absolute seconds** and quote the
search-only residual. (3) **Land iff that residual is negative.**

---

## ★ 0. WHAT LANDED, AND WHAT DID NOT

**Nothing lands in `src/`. The three rungs are dead, killed by the arbiter
rather than by the proxy, and the arbiter itself is repaired.**

| # | result | number |
|---|---|---:|
| 1 | `tools/wallclock_panel.py` had **two asymmetries between the null and the arms** — the null copy lived on **tmpfs** while the binaries under test lived on the NVMe, and runs were **not pinned** while every counter panel since study 45 pins. Both fixed. | null **+0.646 % → −0.051 %**, per-round sd **3.58 % → 1.44 %** |
| 2 | On the repaired instrument the three rungs are **worth nothing**. Startup subtracted in absolute seconds. | **+0.111 % of search time = −0.19 DERIVED Elo**, 95 % CI **[+0.34, −0.50]** |
| 3 | The `.text` cost that justified study 62's rule **does not exist on the shipped PGO binary**, and the same source change costs **+1.43 %** of L1i misses non-PGO. | PGO **−0.09 %** null-corrected vs non-PGO **+1.43 %** |
| 4 | Study 62's ladder **understates these rungs by 1.6–2.3×**; the brief's "+70 KB for −112.98 i/n" pairs a ladder top with a landed win (real: **+37,312 B**, 3.10 i/n per KB). | `shape62d+e` **−14.95** i/n non-PGO, **−21.33** PGO, not −9.28 |
| 5 | **The shape-dispatch vein is closed**, with the census that proves it. The HalfKA make-move fold is already a **100.00 %** cover and nobody had said so. | 4,893,314 calls, three shapes, sums to `hka_into.calls` to the unit |
| 6 | **Three arms built, three measured negative**, all node-identical and eval-hash exact: `hot63` (+0.0001 i/n), the PSQT `as_chunks` hoist (−0.0001 i/n), PSQT const-generic row counts (**+7.12 i/n**). | see §7 |
| 7 | Two tool fixes the study needed on its first hour: `--dump` (a panel used to leave five printed lines, so no successor could recover its sd) and `PGO_PROF_DIR` (`profile-generate` data went to tmpfs = RAM, beside a live GPU trainer). | — |

**And a flag on the series' own headline, §8 item 1: the +3.14 Elo receipt for
the 61+62 redirect was measured on the defective instrument.** Its direction is
sign-test based and survives; its magnitude needs a re-run on the fixed tool,
and that is the single most valuable follow-up in this document.

---

## ★★★ 1. THE ARBITER, RUN AS PRE-REGISTERED — NOT LICENSED, AND THE NULL SAID WHY

`perf-artifacts-63/wallclock-full.txt`, 400 rounds, PGO pair, exactly the
command study 62 §7 item 1 named as its top follow-up:

```
NULL (byte-identical): median +0.646%  sd 4.344  sign 125/400 neg  p=0.0000
ARM                  : median +0.179%  sd 3.580  sign 184/400 neg  p=0.1210
NOT LICENSED: null median +0.646% vs arm +0.179%.
```

**Third refusal in three attempts** (study 62 refused twice), and the third time
the byte-identical null again moved further than the arm — at p < 1e-4. Per the
pre-registered rule, **nothing from this panel is quoted, including "the arm is
zero"**.

But `--dump` means the panel left its raw rounds behind, and a null that reads
p < 1e-4 is a **bug signature, not noise**. Decomposed with
`tools/perf63/diag63.py`:

| decomposition | null median |
|---|---:|
| rotation phase 0 (base at pos 0) | **+0.407 %** |
| rotation phase 1 (base at pos 2) | **+0.661 %** |
| rotation phase 2 (base at pos 1) | **+0.864 %** |
| quarters 1–4 of the run | +0.728 / +0.460 / +0.770 / +0.635 % |
| position medians (seconds) | 3.5243 / 3.5189 / 3.5157 |

**Positive in every phase and every quarter, with the three position medians
inside 0.25 % of each other.** Not position, not drift, not the box: a property
of the null binary. Two asymmetries, both built into the tool since it was
written, both fixed here:

1. **The null copy lived on a different filesystem from the binaries under
   test.** `tempfile.mkdtemp()` lands in `$TMPDIR` = `/tmp`, which on this box
   is **tmpfs** (`huge=advise`, swap-backed) while `bin61/` is on the NVMe. The
   copy now lands beside the base binary. Consistent with the mechanism: the
   1-node startup panel, where almost none of `.text` executes, shows no such
   bias (null 0.2632 s vs base 0.2639 s).
2. **The runs were not pinned.** Every counter panel in the 45–62 chain runs
   `taskset -c "$CPU"`; the wall-clock arbiter never did, so each run migrated
   across a box carrying a GPU trainer at 1.44 CPUs. Measured per-round
   **sd 3.58 %** against **0.98–1.13 %** for study 62's panels — 3.6× the noise
   the arms this tool exists to resolve have. `--cpu` now defaults to core 2.

> **This is what a null is for, and it is the first time in this series that one
> has been believed all the way down to a defect.** Studies 61 and 62 hit the
> same refusal three times between them and each time attributed it to the box
> and asked for a quieter one.

### 1.1 The box was ALSO not quiet, and my own discipline could not see it

Reported because it corrects a claim this study's own artifact made. A
concurrent session on this machine ran a **400-game fixed-node match**
(`d990a1bd`, committed 19:18:30) and a **wall-clock smoke test of its own
against the same PGO pair** (`4fecb7c1`, committed 20:03:14, whose message says
it ran *"while perf study 63 was running its own panel"*) — both inside this
panel's 18:59–20:11 window. My `uptime` + process census ran **before and after**
each panel and is structurally blind to a job that starts and ends between the
two checks. The brief asked for exactly that check and it is not sufficient; the
other session's own answer — sample the busy fraction from `/proc/stat` inside
the tool — is the right one, and it is now merged into `wallclock_panel.py`.

**It does not rescue the panel** (already void) **and it does not explain the
null's bias**: load that inflates all three binaries cancels in a within-round
ratio, which is what interleaving is for, and the bias is uniform across all
four quarters while the intruding jobs are not. But it does mean the
**sd improvement 3.58 % → 1.44 % is CONFOUNDED** between the pinning fix and a
quieter box. Only the null median (+0.646 % → −0.051 %) is cleanly attributable
to the tmpfs fix. Stated as a limitation rather than smoothed over.

---

## ★★★ 2. THE ARBITER, RE-RUN ON THE FIXED INSTRUMENT — the rungs are worth nothing

Same pair, same decision rule, 210 rounds (the box hour the first panel had
consumed was gone; 210 was fixed before the re-run and is what remained).
`perf-artifacts-63/wallclock-full2.txt`:

```
NULL (byte-identical): median -0.051%  sd 1.507  sign 109/210 neg  p=0.6292
ARM                  : median +0.045%  sd 1.437  sign 100/210 neg  p=0.5347
```

**The null is now clean** — median −0.051 %, an even 109/210 split, **p = 0.6292**,
and the per-round sd has fallen 2.5×. Position-corrected it reads −0.066 % at
p = 0.3697. The instrument passes its own test for the first time in this
series at this round count.

And it says the arm is **zero**:

| quantity | value |
|---|---:|
| ARM median | **+0.045 %** |
| se(median) | 0.124 % |
| 95 % CI | **[−0.198 %, +0.289 %]** |
| predicted from −0.6513 % of instructions at 2.08× | −0.313 % |
| distance of the prediction from the measurement | **2.88 se** |

**Startup subtracted in absolute seconds** (the two panel percentages are
relative to different baselines and cannot be subtracted directly), from the
210-round `--startup-frame` panel on the same fixed instrument:

| | median | |
|---|---:|---|
| full frame, base | 3.4770 s | |
| full frame, arm | 3.4809 s | **+3.9 ms** |
| startup frame, base | 0.2537 s | **7.3 %** of the full frame |
| startup frame, arm | 0.2540 s | +0.3 ms |
| **search-only time** | **3.2233 s** | |
| **search-only change** | **+3.6 ms** | **+0.111 %** |

> ### VERDICT: the three rungs are worth **−0.19 DERIVED Elo** (95 % CI
> **[+0.34, −0.50]**) and **NOTHING LANDS**.
>
> Study 62's cut stands. It was reached on a proxy and it is now reached on the
> arbiter, which is a strictly better reason for the same answer. The 7.3 %
> startup share reproduces the handoff's 7.0 % on an independently built pair.

**The interesting part is that the instruction win is real and buys nothing.**
−0.6513 % of PGO instructions, 20/20 at se 0.000, produced +0.111 % of search
time. The 2.08× instructions-overstate-time ratio the handoff measured for
studies 61+62 does **not** transfer. The mechanism is available in study 62's
own §4.2: this engine is **backend-bound 2:1** (9223 dispatch slots per node
lost to the backend against 4531 to the front end), and these rungs remove
instructions from the **tail** — the 7.55 % of threat folds and 20 % of warm
refreshes that are the rarest shapes — which the machine was already absorbing
in spare issue slots. Studies 61 and 62 landed the **modal** shapes (27 % → 92 %
of folds); that is the difference between −0.31 % of measured wall clock and
+0.11 %.

> **`instructions:u` and wall clock diverge exactly where the removed
> instructions were not on the critical path.** That is the generalisation this
> series has been missing, and it is why the per-KB proxy reached the right
> answer for the wrong reason.

---

## ★★ 3. THE COMMISSION'S TABLE IS WRONG BY 1.6–2.3×, IN THE RUNGS' FAVOUR

The brief inherited study 62 §4.1's ladder — `shape62d` −3.91, `shape62e`
−5.37, `dual62b` −13.71 i/n. Re-derived on today's shipped tree (rule 8), every
arm built from `main` at `16098195`, node-identical (**5,001,562**) and eval-hash
exact (**`2e1ede95…79c4`**), whole-frame `instructions:u` reproducible to
**1.5 parts in 10⁸**:

| build | `.text` | i/node | Δ vs base | Δ `.text` | i/n per KB |
|---|---:|---:|---:|---:|---:|
| **non-PGO native** | | | | | |
| `base` | 760,085 | 5467.62 | — | — | — |
| `de` = `shape62e` (both columns) | 773,429 | 5452.67 | **−14.95** | +13,344 | 1.147 |
| `db` = `dual62b` | 779,797 | 5453.91 | **−13.71** | +19,712 | 0.712 |
| `all` = both | 793,141 | 5438.96 | **−28.66** | +33,056 | 0.888 |
| **native + PGO (shipped)** | | | | | |
| `p63-base` | 779,429 | **5335.16** | — | — | — |
| `p63-de` | 792,709 | 5313.83 | **−21.33** | +13,280 | 1.645 |
| `p63-db` | 800,341 | 5321.75 | **−13.42** | +20,912 | 0.657 |
| `p63-all` | 813,573 | 5300.42 | **−34.75** | +34,144 | 1.042 |

`p63-base` reads **5335.16 i/node** against the handoff's shipped **5335.2**, so
the PGO build reproduces the shipped one; `base` reproduces study 62's `.text`
md5 `75c9955e` and its instruction total to 1.5 parts in 10⁸. The two sites are
exactly additive on instructions (−14.95 + −13.71 = −28.66, to the unit).

**`shape62d` + `shape62e` are worth −14.95 i/n together, not −9.28** — **1.61×**
study 62's ladder non-PGO and **2.30×** on the shipped PGO build. `dual62b`
reproduces exactly (−13.71) and loses 2 % under PGO.

The cause is study 62's own §5.4 applied to the BASE rather than the arm: its
ladder measured these two rungs on top of `comb3`, where the landed rings were
**three chained `match` statements**; promotion merged them into **one 33-arm
`match`** and bought −8.64 i/n doing so. Study 62 §8 item 6 wrote the warning —
*"any study that measures arms behind separate `#[cfg]`s and then promotes them
must re-measure the promoted binary"* — and the rungs it dropped are exactly the
arms that warning was about.

> **The rungs were dropped on numbers that understate them by 1.6–2.3×**, and
> they are still dropped by the rule as re-derived (1.645, 0.657, 1.042 against
> 2.0). **The correction does not by itself overturn study 62's decision**, and
> §2 shows the decision was right anyway. A future reader comparing rungs across
> studies must rebuild the base, never read the ladder.

**A second correction to the brief.** It prices study 62's landed win as
"+70 KB of `.text` for −112.98 i/n". The landed set was
**722,773 → 760,085 = +37,312 B (+36.4 KB)** (study 62 §0) = **3.10 i/node per
KB**, comfortably over its own rule. The +70,336 B figure is `comb2`, the TOP of
§4's ladder, which already contains the three rungs this study is about. Pairing
a ladder-top `.text` with a landed instruction win halves the apparent rate of
what actually shipped.

---

## ★★ 4. THE `.text` COST IS A NON-PGO ARTIFACT — the controlled comparison

Study 62's licensed cost measurement, non-PGO, landed set: **L1i misses
+6.199 %, 0/24, p < 0.0001** (null +0.134 %); its `comb2` — this study's arm on
top of that set — read **+9.397 %**.

Both builds of the **same source change**, same counter, same box, same night:

| | PGO (`panel-pgo.txt`, 20 adjudicated rounds) | non-PGO (`panel-nonpgo.txt`, 20 rounds) |
|---|---:|---:|
| reference L1i misses / node | 158.69 | 158.75 |
| **null** | −0.261 % (13/20, p = 0.26) | −0.319 % (18/20, **p = 0.0004 — fails**) |
| `de` | −0.325 % (12/20) → net **−0.06 %** | **+1.025 %** (1/20) → net **+1.34 %** |
| `db` | −0.596 % (17/20) → net −0.34 % | −0.330 % (16/20) → net −0.01 % |
| **`all`** | −0.346 % (14/20, p = 0.12) → net **−0.09 %** | **+1.107 %** (0/20, p = 0.0000) → net **+1.43 %** |

> **+34,144 B of dispatch table costs +1.43 % of L1i misses non-PGO and nothing
> at all on the shipped PGO binary.** PGO's hot/cold splitting moves arms that
> are never taken out of the hot text, so they never enter the instruction
> cache's working set. **Study 62 measured the icache cost on a build the engine
> does not ship**, and the number that motivated its landing rule is a property
> of the non-PGO layout.

Two honesty notes. **(a)** The non-PGO panel's L1i null **fails its own test**
(−0.319 %, 18/20, p = 0.0004), so that column is quoted null-corrected and read
as a bound; the arm at +1.107 % (0/20) is 1.4 points clear of it either way.
**(b)** `cycles:u` is again unresolvable in both panels (PGO: null −0.145 %,
arms +0.239 % to −0.348 %, every p ≥ 0.11; non-PGO: null −0.445 %), reproducing
studies 60 §6, 61 §4.2 and 62 §6.1 for a fourth and fifth time.
`instructions:u` and `branches:u` are 20/20 at se 0.000 against a null of
exactly 0.000 in both.

**This does not license reinstating the rungs.** §2 measured the thing itself
and it is zero; a cost of zero against a benefit of zero is still zero.

---

## ★★ 5. THE DISPATCH VEIN IS CLOSED — the census, so nobody re-mines it

`foldcount` census on the shipped tree (study 62's, re-verified cell by cell:
`dual_pair` reproduces 79.95 % and 95.01 % exactly once the undispatched `(0,0)`
cell is removed, and `thr_pair` sums to `thr_into.calls` to the unit):

| fold site | calls / 5 M nodes | rows | coverage today | with the (dead) rungs |
|---|---:|---:|---:|---:|
| `apply_halfka_deltas_into` (HalfKA make-move) | 4,893,314 | 11,150,925 | **100.00 %** | 100.00 % |
| `apply_threat_deltas_into` (FullThreats) | 4,270,334 | 19,616,896 | 92.45 % | ≈ 97 % |
| `apply_halfka_deltas_dual` (Finny warm refresh) | 823,842 | 4,712,390 | 79.95 % | 95.01 % |
| `apply_halfka_deltas` (Finny cold build) | **128** | 645 | 0 % | 0 % |
| `refresh_psqt_threat` | 98,024 | 407,362 | n/a (from scratch) | n/a |

**The HalfKA make-move fold is already a 100 % cover and nobody had said so.**
`hka_shape.1a1s` 3,529,017 + `1a2s` 1,316,030 + `2a1s` 48,267 = **4,893,314 =
`hka_into.calls` to the unit**, and `apply_dirty_directed` builds both lists in
`[RowOffset; 2]`, so `(0..=2, 0..=2)` is a structural bound rather than an
observation.

So the vein is **exhausted**: one site at 100 %, one at 92 %, one at 80 %, and
the only undispatched site runs 128 times per 5 M nodes. What remains is
uncovered *because it is rare*, which §2 has now shown is precisely the traffic
whose instructions are free. **A successor should not spend another study on
this table.**

---

## ★★ 6. THREE ARMS BUILT, THREE MEASURED NEGATIVE — with their numbers

All node-identical (**5,001,562**) and eval-hash exact; `base` reproduces to
2 parts in 10⁸ across the study, so a whole-frame delta of 10⁶ is far outside
run-to-run variation.

### 6.1 `hot63` — the modal shape hoisted ahead of the 48-arm table

`i8_deltas_into_inner` serves **two** callers and studies 61/62 grew its `match`
from 4 arms to 48 for one of them. The other presents three shapes and is
structurally bounded to nine, so 4,893,314 calls per 5 M nodes walk a two-level
jump table that one predicted branch would decide; `(1,1)` is additionally the
modal THREAT shape (449,864), making it **43.4 % of all 9,163,648 into-folds**.

| build | `.text` | instructions | Δ |
|---|---:|---:|---:|
| `base` | 760,085 | 27,346,634,917 | — |
| `hot` (`hot63`) | **760,085** | 27,346,635,488 | **+571 = +0.0001 i/node** |

**Identical `.text` size and 571 instructions in 27.3 billion**, inside `base`'s
own spread. LLVM already folds the hoisted test into the switch: **the fat
dispatch table has no measurable dispatch cost at its modal shape.** Kept
switchable as the cheapest falsification of "a 48-arm table costs its callers
something".

### 6.2 The HalfKA PSQT fold — the best-looking remaining target, twice negative

`psqt_fold` runs **4,893,314 times per 5 M nodes over 11,150,925 rows** with two
runtime-trip-count loops, and `psqt_add`/`psqt_sub` each re-derive
`table.as_chunks::<BUCKETS>()` per feature — the exact defect **perf study 8
found and fixed in the THREAT channel**, whose event loop hoists `as_chunks`
above the loop and says so in a comment. A fix applied where a defect is found
is not applied where it also applies; this one had stood for five studies.

| arm | Δ instructions | Δ i/node | Δ `.text` |
|---|---:|---:|---:|
| `q0` control (restores today's code) | −632 vs `base` | −0.0001 | 0 |
| `q1` hoist `as_chunks` out of both loops | **−496** | **−0.0001** | +32 B |
| `q2` + compile-time row counts (`psqt_lanes::<NA, NS>`) | **+35,589,740** | **+7.12** | +1,376 B |

**(a) The hoist is a null**, and the reason is instructive: `psqt_fold` is
`#[inline(always)]` with at most two iterations per list, so LLVM already
common-subexpression-eliminates the shift across the inlined calls. The threat
channel's loop is long and unbounded — *that* is the difference, not the
channel. The transferable form of study 8's finding is narrower than its comment
implies.

**(b) Const-generic row counts are 7.12 i/node WORSE** — the same idiom that is
worth −21 to −42 i/node in the i8 kernels one module over. There the loops run
eight times per call (once per tile) over up to 16 rows; here once over at most
two, so the straight-line form buys nothing and pays for copying the indices
into `[usize; NA]`. Study 58's banner, third independent confirmation: *a
loop-control instruction count is not a prize, it is an estimate of what LLVM
would charge you for taking the loop away from it.*

Both arms are reverted; the numbers live in a banner on `psqt_fold` so the next
reader does not rediscover them.

---

## ★ 7. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **The +3.14 Elo receipt for the 61+62 redirect was measured on the
   DEFECTIVE instrument, and it is not re-derived here.** The handoff's panel
   was unpinned with the null on tmpfs. Its null happened to read −0.011 % at an
   even 20/40, so the tmpfs bias was absent or cancelled that time, and its
   arm's **direction** rests on a sign test (29/40, p = 0.0064) which no
   variance defect can manufacture. But its **magnitude** (−1.879 %) carries an
   unpinned se(median) of roughly 0.7 %, not the tight bar "p = 0.0064" reads
   as. **Re-run it on the fixed tool.** It is the single most valuable follow-up
   in this document and it needs one PGO build of `noshape61` plus ~40 minutes:
   `python3 tools/wallclock_panel.py --base <noshape61-PGO> --arm <main-PGO> --rounds 210 --elo --dump ...`
2. **210 rounds, not the pre-registered 400**, on the re-run. The first panel
   consumed the box hour the pre-registration had budgeted. 210 was fixed before
   the re-run and the decision rule was not touched, but the CI is 1.4× wider
   than planned: ±0.24 % rather than ±0.18 %. It is wide enough to exclude the
   predicted −0.313 % at 2.88 se and not wide enough to distinguish +0.111 %
   from 0.
3. **The two rungs are not separated.** `p63-de` and `p63-db` are −0.40 % and
   −0.25 % of instructions; resolving either alone on wall clock needs ~4× the
   rounds of the combined arm. The panel resolved the pair, not the parts. Since
   the pair is zero and both are pure dispatch of the same character, nothing
   turns on it — but it is not established that neither is individually
   positive.
4. **No games**, and none are possible: the arm is node-identical, so a
   fixed-node match is a dead heat by construction and a real-clock match would
   need to resolve 0.1 %.
5. **The PGO-vs-non-PGO L1i mechanism is inferred, not instrumented.** §4
   measures that the cost is present non-PGO and absent PGO; it does not show
   the hot/cold split in the layout. A `nm`/section-layout diff of the two
   binaries would settle it and was not run.
6. **Was the CORRECTED panel quiet?** It cannot be verified retroactively — the
   box-load check (`4fecb7c1`) landed in `main` while that panel was already
   running, so no busy fraction was recorded for it. The before/after process
   censuses at 20:16:08 and 20:52:40 are identical (GPU trainer + browser), the
   concurrent session's commits inside that window are documentation and log
   analysis rather than jobs, and summing the census's top-five CPU shares gives
   **≈ 13.3 % busy against the new tool's 15 % threshold** — under it, but not
   comfortably. The panel's own null passing at 109/210, p = 0.6292 is the
   strongest evidence available and it is indirect.
7. **`cycles:u` remains unresolvable on this box**, in both builds, for the
   fifth study running. Wall clock with a pinned, same-filesystem null is now
   the better cycle-scale instrument, which is a reversal of the assumption the
   52–62 chain worked under.

---

## ★ 8. WHAT THE NEXT STUDY SHOULD KNOW

1. **A refusal from `wallclock_panel.py` is a hypothesis about the TOOL, not
   about the box.** Three refusals were attributed to load; the cause was two
   asymmetries in the harness. `--dump` + `tools/perf63/diag63.py` turn a
   refusal into a diagnosis in one command: if the null's excursion is present
   in every rotation phase and every quarter, it is not the box.
2. **Instructions and time diverge where the instructions are off the critical
   path.** −0.6513 % of PGO instructions bought +0.111 % of search time here,
   against 2.08× for studies 61+62. Those landed the MODAL fold shapes; these
   covered the tail. On a machine that is backend-bound 2:1, removing
   instructions the machine was absorbing in spare slots is worth nothing.
   **Before pricing a dispatch rung, ask what fraction of TRAFFIC it converts,
   not how many instructions it deletes.**
3. **Measure `.text` costs on the build you ship.** The +6.2 %/+9.4 % L1i rise
   that produced study 62's landing rule is a non-PGO artifact; PGO shows
   −0.09 %.
4. **The fold shape-dispatch vein is closed** (§5), and the two `#[cfg]` levers
   `shape62d`/`shape62e`/`dual62b` can now be deleted by anyone who wants the
   `Cargo.toml` shorter — they have a wall-clock verdict, not just a proxy one.
5. **`psqt_fold` is not a target** (§6.2) and the banner on it says why with
   numbers, so nobody spends another evening on the most attractive-looking
   fold in the engine.
6. **Still open, inherited and untouched:** study 43 §9 item 2's subtractive
   Finny-line arm; study 57 §6's FNV-1a-64 payload checksum (~45 % of startup,
   needs a format ruling); study 48 §9 item 3's AVX2 `activate` hoist. Study 61
   §3's capacity closure stands.

---

## ★ 9. GATES

| gate | result |
|---|---|
| node identity, study frame, every binary (12 built) | **5,001,562** — exact |
| wl50 eval hash, every binary | **`2e1ede95…79c4`** — exact |
| byte-identical null, PGO counter panel | `instructions:u` **+0.000 %**, `branches:u` **+0.000 %**, both se 0.000 |
| byte-identical null, non-PGO counter panel | `instructions:u` **−0.000 %**, se 0.000 |
| byte-identical null, FIXED wall-clock panel | **−0.051 %, 109/210, p = 0.6292** |
| `base` reproduces study 62's landed binary | `.text` **760,085** and md5 **`75c9955e`**, both exact; instructions to **1.5 parts in 10⁸** |
| `p63-base` reproduces the SHIPPED PGO engine | **5335.16 i/node** vs the handoff's 5335.2 |
| realised mean panel position | **2.00** for all five labels, both counter panels |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | **exit 0 — `All checks passed.`, `Bench: 587683`, eval hash `2e1ede95…79c4`, d13 nodes `1739588`; all three pins unmoved** |
| `cargo test --release -- --ignored perft_full_suite` | **ok** (1 passed) |
| `cargo clippy --all-targets` / `cargo fmt` | clean / no change |

| rebase re-gate (study 56's `bcmisc55` rule) | `main` moved `16098195 → d84c7e44` **during the study**, and a concurrent session edited `tools/wallclock_panel.py` in that span (`4fecb7c1`, a box-load check). Rebased and both changes merged by hand; `git diff --stat 16098195 d84c7e44` touches **no `src/` and no `Cargo.toml`**, so no measurement expired. The merged tool smoke-tested end to end. |

D24 perft ritual not strictly required (nothing under `movegen`, `board`
mutation, `zobrist` or `attacks` is touched) — run anyway. The gate was run on
the final tree: `hot63` (a `#[cfg]`-off feature) plus doc comments, no
behavioural change, which is why every pin is unmoved.

---

## ★ 10. REPRODUCE

```bash
ln -s /path/to/chessengine/data data          # the worktree needs the net

# the non-PGO lattice
bash tools/perf61/build61.sh base
bash tools/perf61/build61.sh de   shape62e
bash tools/perf61/build61.sh db   dual62b
bash tools/perf61/build61.sh all  shape62e,dual62b
P61_CPU=2 bash tools/perf61/ident61.sh base de db all

# the PGO lattice (PGO_PROF_DIR keeps profile-generate data off tmpfs)
PGO_PROF_DIR=/var/tmp/s63 CARGO_BUILD_JOBS=2 \
  bash tools/build_pgo.sh sota,arch-d48x p63-base "$(readlink -f engines/canonical/net.cenn)"
#   ... and the same for shape62e / dual62b / shape62e,dual62b

# THE ARBITER, on the FIXED instrument (pinned, null beside the base binary)
python3 tools/wallclock_panel.py --base bin61/pbase --arm bin61/pall \
  --rounds 210 --elo --dump docs/research/perf-artifacts-63/wc-full2-rounds.tsv
python3 tools/wallclock_panel.py --base bin61/pbase --arm bin61/pall \
  --rounds 210 --startup-frame --dump docs/research/perf-artifacts-63/wc-startup2-rounds.tsv
python3 tools/perf63/diag63.py docs/research/perf-artifacts-63/wc-full2-rounds.tsv

# the L1i contrast, PGO and non-PGO
P61_CPU=2 P61_EV='instructions:u,branches:u,ic_tag_hit_miss.instruction_cache_miss,cycles:u' \
  P61_PANEL=docs/research/perf-artifacts-63/panel-pgo.txt \
  bash tools/perf61/panel61.sh 24 pbase pnull pde pdb pall
python3 tools/perf61/med61.py docs/research/perf-artifacts-63/panel-pgo.txt pbase pnull pde pdb pall

CHECK_NNUE_GATE=1 bash tools/check.sh
cargo test --release -- --ignored perft_full_suite
```

**Do not run a panel while anything else builds**, and do not park perf text on
`/tmp` — it is tmpfs, which is also what §1 defect 1 turned out to be.
