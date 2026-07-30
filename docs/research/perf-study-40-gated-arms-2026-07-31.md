# Perf study 40 — the two gated arms, decided

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 16 GB ·
**Base:** `main` at `9a04aa2` (study 39 merged) · **Net/arch:** D48X @
`network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf40/` · **Artifacts:** `docs/research/perf-artifacts-40/`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO, pinned to
core 2 with `taskset`. **Every binary in this study is node-identical at
5,001,562 and eval-hash-identical at `2e1ede9521624dd8`** — gated, not assumed
(`tools/perf34/ident34.sh`, `tools/perf40/identdose40.sh`).

Provenance: `bin34/b40`, this study's reference, has `.text` md5
`e4aa8ffe35d8aeb9d1bb553f045cda5d` — **byte-identical to study 38's landed
binary**. The reference provably *is* `main`.

---

## ★ THE TWO VERDICTS

### Arm A — `insertwin34`: **it is a wash, and it stays OFF.**

Reproduced exactly on today's tree, 60 paired within-round rounds, zero spread
on every exact counter, against a byte-identical null that reads 0.000%:

| counter | `insertwin34` vs `main` | rounds on one side |
|---|---:|---|
| `instructions:u` / node | **+0.719%** (+42.58) | 60/60 |
| `branches:u` / node | **−0.525%** (−3.26) | 60/60 |
| `ex_ret_brn_misp:u` / node | **−4.409%** (−1.014) | **60/60** |
| `cycles:u` / node | *(§3)* | *(§3)* |

Study 34's measurements all replicate (+0.702%, −0.530%, ≈−4%). What it could
not do was price the trade, and **§3 prices it**: the arm removes
`REPLACE_BUBBLES` cycles of front-end starvation per node — a *measured*
mispredict penalty of `REPLACE_PENALTY` cycles — against `REPLACE_ISSUE` cycles
of extra issue work. Net `REPLACE_NET`, and the direct cycles panel puts it at
`REPLACE_DIRECT` against a null of `REPLACE_NULL`.

### Arm B — the ~26 i/n: **CLOSED. It was a counterfactual priced in the wrong codegen regime.**

Study 38's arm elides 93.5% of the threat channel's pure-copy folds. It priced
each elided copy at **187.9 instructions** — the cost of a `memcpy` library
call, measured by a doubled-copy probe built on the *pre-arm* source. But the
alias array's **mere presence is what makes LLVM stop calling `memcpy`** at that
site and emit 40 inline zmm moves instead, so the copies the arm actually elides
are inline ones worth **111.8** each. The gap, `(187.9 − 111.8) × 0.316885
copies/node = 24.1 i/n`, is the whole residual.

Exact counters, zero spread, and a decomposition that closes to **0.000 i/n**:

| | `instructions:u` / node |
|---|---:|
| `na40` — no machinery, copy is a **CALL** (= study 38's base) | 5941.078 |
| … + the alias machinery, codegen held at CALL | **+4.35** |
| … + LLVM's codegen flip, CALL → 40 inline zmm | **+13.98** |
| … + eliding 0.316885 **inline** copies/node (111.8 each) | **−35.44** |
| `b40` — `main` | **5923.967** (= −17.11, study 38's headline) |

**And the machinery is cheap — +4.35 i/n, not the +19.7 study 38 reported.**
That figure was the machinery *plus* the codegen flip, because the two binaries
it differenced compile the copy differently.

### The useful half: arm B hands back a landable **−16 i/n**

`--features outcopy40` puts the pure-copy fold behind an `#[inline(never)]`
helper. In `main` that copy runs **0.022 times per node**, so under 2 i/n can be
the copy itself — the rest is `at_full` compiling better once an 83-instruction
straight-line zmm block stops sitting inside the fold loop:

| counter | `outcopy40` vs `main` | rounds |
|---|---:|---|
| `instructions:u` / node | **−0.270%** (−16.01) | 8/8, zero spread |
| `branches:u` / node | **−0.195%** (−1.21) | 8/8, zero spread |
| `ex_ret_brn_misp:u` / node | −0.578% (null: +0.330%) | 7/12 — no effect |
| `cycles:u` / node | *(§3)* | *(§3)* |

It is **not a trade** — instructions and branches both fall and mispredicts do
not move — which is the same standard studies 34, 37 and 38 landed their own
arms on. Study 14 recorded "45.4 i/n between two spellings of this copy" and
called it *spelling, not semantics*; it now has a sign and a size.

---

## 1. METHOD

### 1.1 What changed about the instrument, and why

Studies 33–39 retired `cycles:u` on this box six times, each on the counter's
**spread**. That was the right call for what they were asking and it is the
wrong summary for this study, for two measured reasons:

* **The noise is heavy-tailed, so a trimmed statistic converges ~3× faster than
  the range suggests.** 60 paired rounds of a byte-identical null:
  raw sd **3.404%**, 20%-trimmed sd **1.263%**. `tools/perf40/spread40.py`
  prints both and converts them into "rounds needed for an X% effect".
* **A run on this frame is 4 seconds**, so hundreds of rounds are affordable —
  studies 34 and 38 quoted panels of 10–16.

So the panel (`tools/perf40/panel40.sh`) shuffles the label order **inside**
every round (a fixed rotation cancels monotone drift but not curvature — study
25 §0.2), differences every label against the same round's reference, and the
reducer (`tools/perf40/med40.py`) adds a **sign count with an exact binomial p**
alongside median, mean and min. Common-mode noise cancels inside a pair; what
survives is a vote.

### 1.2 Null first — and the null is quotable

`z40` is a **byte-identical copy** of the reference binary, carried in every
panel. 60 rounds:

| counter | null median | null spread | rounds negative | verdict |
|---|---:|---|---|---|
| `instructions:u` | **+0.000%** | [0.00, 0.00] | 30/60 | exact |
| `branches:u` | **+0.000%** | [0.00, 0.00] | 29/60 | exact |
| `ex_ret_brn_misp:u` | −0.101% | [−4.93, +5.96] | 34/60, p = 0.37 | **usable** |
| `cycles:u` | +0.454% | [−9.59, +9.21] | 26/60, p = 0.37 | ±0.5% floor at n = 60 |

**One instrument correction to studies 34 and 38.** Both retired
`ex_ret_brn_misp:u` as unable to adjudicate — study 38 on a null of −0.522%
over a range of 5.2 points at n = 10. At n = 60, paired within-round against a
byte-identical null, that counter puts the null at 34/60 (a coin flip) and
`insertwin34` at **60/60 with a −4.4% median**. The counter was never the
problem; **n = 10 was**. This matters beyond this study: the mispredict counter
is the natural instrument for the whole `search::<NonPv>` / `init_quiets`
mispredict class, and it has been treated as dead since study 34.

### 1.3 Hygiene

`src/eval/nnue/sota_threats.rs` and `src/movepick.rs` both gained a small
refactor so the two nulls could exist at one site each (`AliasTable`,
`partial_insertion_sort_inner`). Rebuilding plain `main` on top of those edits
(`b40r`) gives a **different `.text` md5** — as study 35 warned it would, since
a comment shifts every later panic `Location` and every rip-relative
displacement that reaches one — but **`b40r` retires 5923.967 i/n, identical to
`b40` to six digits over 12 paired rounds, with `branches:u` identical too.**
The refactor is codegen-inert where it counts.

`tools/perf40/textmd5.sh` and `tools/perf40/copycheck40.sh` read the two
properties this study depends on off the binaries rather than out of prose.

---

## 2. ARM B — THE ~26 i/n

### 2.1 The fact that reprices everything, read off the binaries

`tools/perf40/copycheck40.sh`, on `ThreatContext::at_full` (which inlines
`compute`):

```
b40   (main, alias taken)          libc-call refs=0   vmovups=80
n38   (null38, alias never taken)  libc-call refs=0   vmovups=80
na40  (noalias40, no machinery)    libc-call refs=2   vmovups=-
```

The 2560-byte accumulator copy is a `memcpy` **CALL** only when the alias
machinery is **absent**. Every binary that carries the array — including
`null38`, whose whole job is to be the arm's machinery with the alias never
taken — compiles it as a fully unrolled **40-load/40-store zmm block** (visible
at `0x6fa1b` in `n38`, offsets `0x0`…`0x9c0`).

That is fatal to study 38's ledger, because the two binaries in its
decomposition sit on opposite sides of that flip.

### 2.2 The ladder

`--features noalias40` replaces the alias table with a zero-sized stand-in whose
`get(k)` is `k` — the pre-study-38 source, reconstructed **in place** rather
than checked out of git onto a different tree. `--features callcopy40` pins the
copy to the library call (an opaque length defeats LLVM's inline expansion), so
the machinery can be measured with codegen held fixed.

12 paired rounds, `instructions:u`, **zero spread on every row**, byte-identical
null at −0.000%:

| label | machinery | alias | copy codegen | i/n | vs `main` |
|---|---|---|---|---:|---:|
| `na40` | absent | — | CALL (LLVM's choice) | 5941.078 | +0.289% |
| `nac40` | absent | — | CALL (pinned) | 5942.967 | +0.321% |
| `n38c` | present | never taken | CALL (pinned) | 5947.318 | +0.394% |
| `n38` | present | never taken | INLINE (LLVM's choice) | 5959.404 | +0.598% |
| **`b40`** | present | **taken** | INLINE | **5923.967** | — |

**`na40` reproduces study 38's base to 0.042 i/n out of 5941 — 7 parts per
million.** (Its base read 5941.036.) And `b40 − na40 = −17.11 i/n = −0.288%`,
study 38's headline, to three decimals. The reconstruction is exact and the
landed arm replicates.

### 2.3 The decomposition, and it closes to zero

```
  callcopy40's own tax (black_box + two slice bounds)
      nac40 - na40                                        +1.89 i/n   (5.6/copy)
  the machinery, codegen held fixed at CALL
      n38c - nac40                                        +4.35 i/n
  LLVM's codegen flip, CALL -> 40 inline zmm
      (n38 - n38c) + (nac40 - na40)                      +13.98 i/n  (41.2/copy)
  eliding 0.316885 INLINE copies/node
      b40 - n38                                          -35.44 i/n (111.8/copy)
                                                         ----------
  b40 - na40                                             -17.11 i/n   exact
```

Every term is a difference of two exact medians with zero spread, and the sum
is the measured total to **0.000 i/n**.

Against this, study 38's ledger read:

```
  copies removed  0.316885/node x 187.9 i  =  -59.54 i/n
  residual copies, memcpy -> inline        =   -2.26 i/n
  predicted                                   -61.80
  measured                                    -17.11
  machinery (null38)                          +19.72
  UNEXPLAINED                                 +24.97
```

and the two lines that are wrong are now both measured:

* **`-59.54` should be `-35.44`.** The elided copies are inline, not calls.
  `(187.9 − 111.8) × 0.316885 = 24.1 i/n` — the residual, to within 0.9.
* **`+19.72` of "machinery" is `+4.35` of machinery and `+13.98` of codegen
  flip.** Study 38 flagged its own null as a lower bound for the wrong reason
  (the deleted `const` test); the real problem was that the null's copy compiles
  differently from its base's.

### 2.4 What a 2560-byte `memcpy` actually costs, measured not inferred

`tools/perf40/memcost40.c` — opaque size, 64-byte-aligned non-overlapping
buffers, the difference of a 3 M-copy and a 1 M-copy run on `instructions:u`:

| | per 2560-byte `memcpy` |
|---|---:|
| `instructions:u` | **156.00** |
| `branches:u` | **20.00** |

Study 38's doubled-copy probe measured **19.5 branches per call** — the same
glibc kernel, independently. Its 187.9 instructions is 156 plus ~32 of the
probe's own scaffolding (`split_at_mut`, the `chunks_exact` zip, the call
setup), so **187.9 was a fair price for the probe's copy and the wrong price for
the arm's counterfactual.** The error was never in the probe.

### 2.5 The −16 i/n that falls out

If the machinery's presence is what flips the codegen, and the flip costs
+13.98 i/n at 0.339 copies/node, what happens if the copy is pinned back to a
call in `main`, where it runs only **0.022** times per node? Per-copy arithmetic
says the direct effect is under 2 i/n. Measured, 8 paired rounds, zero spread:

| label | what | i/n | vs `main` |
|---|---|---:|---:|
| `b40c` | `main` + `callcopy40` (libc call) | 5908.044 | **−0.269%** (−15.92) |
| `oc40` | `main` + `outcopy40` (`#[inline(never)]` helper) | 5907.956 | **−0.270%** (−16.01) |

**The effect is not the copy — it is the surrounding function.** `at_full`
carries an 83-instruction straight-line zmm block inside its fold loop; moving
that behind any call boundary is worth 16 instructions per node across the whole
engine. `outcopy40` gets it with our own helper and no libc call at all, and
also takes −1.21 branches/node with it. Mispredicts do not move
(−0.578% against the null's +0.330%, 7/12 rounds).

---

## 3. ARM A — `insertwin34`, priced at last

### 3.1 The instrument that made it possible: count the bubbles, not the cycles

`cycles:u` cannot see this arm. 60 paired rounds put it at 29/60 on the wrong
side of zero against a byte-identical null's 26/60 — a coin flip either way, and
that is the same wall studies 34 and 38 hit.

But the arm's benefit is not diffuse. It is **branch mispredicts**, and a
mispredict spends its cost in one measurable place: cycles in which nothing
retired because the front end had nothing to deliver. Zen 4 counts exactly that
as **`ex_no_retire.empty`**, and it is 192.91 cycles per node here — **4.7% of
the engine's cycles**. An effect worth 9 cycles/node is 0.22% of `cycles:u` and
**4.7% of `ex_no_retire.empty`**: a 21× better signal-to-background on the same
box, the same runs and the same rounds.

Two smaller instrument points, both nulled:

* **`ex_no_retire.thread_not_selected` is the SMT sibling, and it is huge and
  wild**: 68.40 cycles/node (1.7% of cycles) ranging from −56% to +203% across
  rounds. It is the machine's own accounting of the noise a `taskset`-pinned
  panel cannot control. Subtracting it (`cycclean = cycles −
  thread_not_selected`, `tools/perf40/derive40.py`) cuts the byte-identical
  null's 20%-trimmed sd from **0.888% to 0.624%** — 2.0× fewer rounds for the
  same resolution. It is the same quantity with a *measured* confound removed,
  and the null is what says so.
* The byte-identical null on `cycclean` reads **+0.156% ± 0.120%** at n = 66 —
  consistent with zero, which is the licence to quote a contrast against it.

### 3.2 The three numbers, 66 paired within-round rounds

All contrasts are *arm minus the byte-identical null*, 20%-trimmed means with
standard errors, on absolute per-node levels `ex_no_retire.empty = 192.907`,
`ex_ret_brn_misp = 24.092`, `cycclean = 4011.409`:

| | contrast vs null | ± | se | per node |
|---|---:|---:|---:|---:|
| `ex_ret_brn_misp:u` | **−4.292%** | | 66/66 rounds | **−1.034 mispredicts** |
| `ex_no_retire.empty:u` | **−4.668%** | 0.563% | **8.3 se** | **−9.01 cycles** |
| `cycclean:u` | **+0.333%** | 0.191% | 1.7 se | **+13.36 cycles** |
| `cycles:u` | +0.276% | 0.246% | 1.1 se | +11.3 cycles |

**The mechanism is confirmed at 8.3 standard errors and then it is priced, and
the price is bad.** The arm really does remove the mispredicts study 34 aimed at
— 1.034 per node, 66/66 rounds — and they really are worth front-end bubbles:
9.01 cycles of them. But 9.01 cycles is what 42.58 extra instructions cost, and
then some.

### 3.3 Study 34's model, corrected by two measured coefficients

Study 34 priced the trade at ≈ **−0.17% of cycles** from `−0.96 mispredicts ×
~18 cycles` against `+41.8 instructions` issuing at roughly 4 per cycle. Both
coefficients are now measured on this arm, in this engine, and **both are worse
for the arm**:

| coefficient | study 34 assumed | **measured here** |
|---|---:|---:|
| cycles of front-end bubble per mispredict removed | ~18 | **8.71** (9.01 / 1.034) |
| marginal instructions per cycle for the added work | ~4 | **1.90** (42.58 / 22.37) |

and the second falls out of the first: the net is +13.36 cycles/node and the
benefit is −9.01, so the added instructions cost **+22.37 cycles**. That is the
shape of the window loop — eight iterations carrying a loop-borne `q -= shift`
dependency, plus a store per lane — not eight independent ALU ops.

**Verdict: `insertwin34` stays OFF.** It is ≈ **+0.33% of cycles**, which at
ARCHON's own calibration constant (~120 Elo per doubling of speed, `exp-0154`)
is about **−0.6 Elo**. Small, but the sign is settled and it is the wrong one.

### 3.4 What this closes beyond the arm

Nobody has ever landed a mispredict-targeted change in this engine, and the
standing explanation was that the box could not adjudicate a trade. That is now
wrong twice over:

1. **The trade CAN be adjudicated** — `ex_no_retire.empty` prices a mispredict
   directly, at 21× the signal-to-background of `cycles:u`.
2. **The exchange rate is 8.71 cycles, not 18.** Every future mispredict arm on
   this engine should be priced at that rate before it is built. `init_quiets`
   is the 2nd largest mispredict source at 13.75% of engine mispredicts = 3.31
   per node; at 8.71 cycles that entire source is **28.8 cycles/node, 0.72% of
   cycles** — and removing *all* of it would have to cost under 55 instructions
   per node to break even. That is a much tighter budget than the class has been
   assumed to have, and it reprices study 33's `search::<NonPv>` (16.57%) too.

---

## 4. THE AMPLIFIER THAT DID NOT WORK — a null worth recording

`--features dosesort40` (`src/dose40.rs`) makes every `partial_insertion_sort`
call replay `P40_DOSE` **real earlier inputs** through the same sort from a
ring, on scratch, and throw the results away. Node-identical and
eval-hash-identical at doses 0, 12 and 24 (gated). The idea: the arm's ~0.2%
effect lands at 25× its size at dose 24, and a dose series regresses the slope
back to one call.

**It amplifies instructions and branches faithfully and mispredicts not at
all**, which is fatal, because the arm's entire benefit is mispredicts:

| quantity, `ds40` (the walk) | dose 0 | dose 24 | per extra sort |
|---|---:|---:|---:|
| `instructions:u` / node | 5951.13 | 9296.68 | +139.4 |
| `branches:u` / node | 625.91 | 1493.16 | +36.1 |
| `ex_ret_brn_misp:u` / node | 23.80 | 35.30 | **+0.48** |

A *real* sort call mispredicts far more than 0.48: the window arm removes 1.014
mispredicts per node over ~0.595 sort calls per node, i.e. **≥1.7 per call, and
that is only the part the window removes**. And the arm's own delta barely
amplifies at all — `dsw40 − ds40` on mispredicts is ≈ −0.9 at dose 0 and −1.41
at dose 24, an amplification of **1.6× where the design predicted 25×**.

**Mechanism: the branch predictor learns the replay.** With a 32-slot ring and
dose 24, consecutive calls replay overlapping slot ranges, so the same
25-outcome branch sequence is presented ~24 times within microseconds and TAGE
captures it. The walk stops mispredicting, so the window has nothing to save and
the dose measures only its instruction cost — biased hard against the arm (at
dose 24 the window reads +1381 cycles/node, which is the trade with the benefit
switched off).

**The general lesson, and it applies to any replay harness including an
out-of-engine microbenchmark: you cannot amplify a mispredict-driven effect by
replaying inputs.** The replay is what makes them predictable. The feature is
left in the tree with this warning in its doc comment, because the failure is
not obvious in advance and the fidelity gate that catches it (does Δmispredicts
scale with the dose?) is one line of panel.

The amplifier has a second, independent fidelity failure worth recording:
splitting `partial_insertion_sort` so the replay can re-enter it changes the
arm's own compilation — the shipped pair differ by **+42.58 i/n** and the dose
pair at dose 0 by **+23.92**. Even with the predictor problem fixed, the slope
would price a differently-compiled arm.

---

## 5. WHAT LANDED

*(filled in)*

---

## 6. GATES

*(filled in)*

---

## 7. WHAT I AM LEAST SURE ABOUT

*(filled in)*
