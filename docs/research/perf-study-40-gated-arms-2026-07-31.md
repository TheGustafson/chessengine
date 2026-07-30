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

### Arm A — `insertwin34`: **it LOSES, ≈ +0.19% of cycles. It stays OFF.**

Reproduced exactly on today's tree, 60 paired within-round rounds, zero spread
on every exact counter, against a byte-identical null that reads 0.000%:

| counter | `insertwin34` vs `main` | rounds on one side |
|---|---:|---|
| `instructions:u` / node | **+0.719%** (+42.58) | 60/60 |
| `branches:u` / node | **−0.525%** (−3.26) | 60/60 |
| `ex_ret_brn_misp:u` / node | **−4.409%** (−1.014) | **60/60** |

Study 34's measurements all replicate (+0.702%, −0.530%, ≈−4%). What it could
not do was **price** the trade, because `cycles:u` cannot resolve 0.2% here.

**§3 prices it, on a counter that can.** A mispredict spends its cost as
front-end starvation, and Zen 4 counts exactly that as
`ex_no_retire.empty` — 192.91 cycles/node, **4.7% of the engine**, so a
9-cycle effect is 4.7% of *it* against 0.22% of `cycles:u`. 159 paired rounds,
contrasts against a byte-identical null:

| | contrast vs null | se | per node |
|---|---:|---:|---:|
| `ex_ret_brn_misp:u` | **−4.517%** | **44.3** | −1.088 mispredicts |
| `ex_no_retire.empty:u` | **−5.150%** | **14.4** | **−9.93 cycles bought** |
| `cycles` − SMT | **+0.186%** | 1.4 | **+7.47 cycles net** |

**The arm buys 9.93 cycles of mispredict bubble and pays 17.40 for the 42.58
instructions.** Study 34 assumed ~18 cycles per mispredict and ~4 IPC on the
added work; measured, they are **9.13** and **2.45** — both worse for the arm,
and together they are the whole gap between its predicted −0.17% and the
measured **+0.19%** (≈ −0.3 Elo).

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

### The useful half: arm B hands back a **−16 i/n arm, and it LANDED**

Putting the pure-copy fold behind an `#[inline(never)]` helper
(`copy_acc_outlined`) is worth 16 instructions per node. In `main` that copy
runs **0.022 times per node**, so under 2 i/n can be the copy itself — the rest
is `at_full` compiling better once an 83-instruction straight-line zmm block
stops sitting inside the fold loop:

| counter | vs `main` | evidence |
|---|---:|---|
| `instructions:u` / node | **−0.270%** (−16.01) | 8/8 rounds, zero spread |
| `branches:u` / node | **−0.195%** (−1.21) | 8/8 rounds, zero spread |
| `ex_no_retire.empty:u` / node | +0.325% ± 0.342% | no effect (0.9 se) |
| `ex_ret_brn_misp:u` / node | −0.407% ± 0.096% (−0.098) | 4.2 se |
| `cycles:u` / node | **−0.415% ± 0.179%** | 2.3 se, n = 159 |
| `cycles` − SMT / node | **−0.300% ± 0.152%** (−12.05) | 2.0 se, n = 159 |

It is **not a trade** — instructions and branches both fall, front-end
starvation does not move, and the cycles contrast is the right sign at 2 se.
That is the standard studies 34, 37 and 38 landed their own arms on, with a
cycles number none of them had. Study 14 recorded "45.4 i/n between two
spellings of this copy" and called it *spelling, not semantics*; it now has a
sign and a size.

### What this study also hands to the next one

* **`ex_no_retire.empty` is the instrument for the whole mispredict class**, and
  the exchange rate on this engine is **9.13 cycles per mispredict**, not the
  ~18 everyone has been assuming. At that rate `init_quiets`' entire mispredict
  share (13.75%, 3.31/node) is **28.8 cycles/node = 0.72% of cycles**, and
  `search::<NonPv>`'s (16.57%) is 0.87%. Any future mispredict arm has to cost
  under ~4.3 instructions per mispredict removed to break even.
* **`ex_ret_brn_misp:u` is not dead.** Studies 34 and 38 retired it at n = 10.
  At n = 159, paired within-round against a byte-identical null, it puts the
  null at a coin flip and this arm at **44 standard errors**.
* **`ex_no_retire.thread_not_selected` is the SMT sibling**, 68.4 cycles/node
  swinging −56% to +203% across rounds. Subtracting it is worth 2× in rounds.
* **You cannot amplify a mispredict effect by replaying inputs** (§4) — the
  replay is what makes them predictable.
* **Never `git stash` in a worktree** (§6.1). It cost time here and it cost
  perf study 24 time before that.

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

**The three rates were re-derived on this tree, not inherited** (rule 8). Two
`foldcount` builds, both node- and eval-hash-identical to `main`:

| | `fcn40` (`noalias40`) | `fc40` (`main`) |
|---|---:|---:|
| `thr_copy.calls` | **1,694,891** = 0.338871/node | **109,963** = 0.021986/node |
| `thr_alias.calls` | 0 | **792,464** |
| `thr_joint.n11e11` (both perspectives empty) | 792,464 | 792,464 |

Elided: 1,694,891 − 109,963 = **1,584,928 = 0.316885/node**. Every figure
matches study 38's census exactly, and the `thr_joint` histogram is bit-identical
between the two builds — which is the check that `noalias40` changes only how the
copy is *performed*, never which plies need one.

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
| `oc40` | `main` + `copy_acc_outlined` (`#[inline(never)]` helper) | 5907.956 | **−0.270%** (−16.01) |

**The effect is not the copy — it is the surrounding function.** `at_full`
carries an 83-instruction straight-line zmm block inside its fold loop; moving
that behind *any* call boundary is worth 16 instructions per node across the
whole engine, and the two spellings agree to 0.09 i/n. The outlined helper gets
it with our own function and no libc call at all, and takes −1.21 branches/node
with it. Over the 159-round panel its front-end starvation does not move
(+0.325% ± 0.342%, 0.9 se), its mispredicts fall a little (−0.407% ± 0.096%,
−0.098/node) and **its cycles fall: −0.415% ± 0.179% raw, −0.300% ± 0.152%
SMT-corrected** — 12.05 cycles/node, which is 16.01 instructions at a marginal
1.33 per cycle.

**This is what landed** (§5), as `copy_acc_outlined`. Rebuilt from the landing
tree it retires **5907.956 i/n and 619.310 br/n — identical to `oc40` to six
digits** over 6 paired rounds, node- and eval-hash-identical.

### 2.6 `#[inline(never)]` is not a lever — study 26 tried it and LOST

Perf study 26 §6.1 outlined the AVX-512 `l2_tail` kernel for a similar reason
(the codegen it forced was demonstrably better — 17 clean `vbroadcastss` in
place of a 36-instruction shuffle chain) and it **lost 4.26 i/n**, "because the
caller then spills its live state around the call". Reverted.

Same tool, opposite sign, and the difference is not luck — it is the **call
rate**:

| | `l2_tail` (study 26) | this copy |
|---|---:|---:|
| calls per node | ~0.43 (every evaluation) | **0.022** |
| caller pays spills around the call | **every eval** | 1 node in 45 |
| measured | **+4.26 i/n** | **−16.01 i/n** |

Outlining moves work from the caller's straight-line body into a callee, and
charges a spill/reload at every call. It wins where the block is **large and
the call rate is low** — which is exactly this site after study 38's alias
elided 93.5% of the copies, and exactly not `l2_tail`. **The two results
together are the rule**, and either alone would be a bad generalisation.

---

## 3. ARM A — `insertwin34`, priced at last

### 3.1 The instrument that made it possible: count the bubbles, not the cycles

`cycles:u` cannot see this arm. 60 paired rounds put it at 29/60 on the wrong
side of zero against a byte-identical null's 26/60 — a coin flip either way, and
that is the same wall studies 34 and 38 hit.

**Perf study 36 §6.1 was right about that and right to refuse.** It was
commissioned to adjudicate this arm, declined, and said why: "the trade is
0.17%, this box's cycles floor is 9.8 to 14.2 points of spread around a median
that itself moved +0.216% to +1.269%. That is 30–250× the effect… The
instrument does not exist tonight." Every word of that is confirmed here. Its
conclusion — "it needs a genuinely idle box and ~40 paired rounds" — is the one
thing that turned out wrong, and not in the direction of more rounds: **159
rounds of `cycles:u` still only reach 1.4 se on this arm.** What was needed was
not a quieter box, it was a different counter.

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
* The byte-identical null on `cycclean` reads **+0.282% ± 0.095%** at n = 159 —
  small but not zero, which is exactly why every number in §3.2 is quoted as a
  *contrast against it* rather than raw. Read alone, the arm's `cycclean` delta
  is +0.468%; two thirds of that is the panel, not the arm.

### 3.2 The four numbers, 159 paired within-round rounds

All contrasts are *arm minus the byte-identical null*, 20%-trimmed means with
standard errors, on absolute per-node levels `ex_no_retire.empty = 192.907`,
`ex_ret_brn_misp = 24.092`, `cycclean = 4017.871`:

| | contrast vs null | ± | se | per node |
|---|---:|---:|---:|---:|
| `ex_ret_brn_misp:u` | **−4.517%** | 0.102% | **44.3 se** (158/159 rounds) | **−1.088 mispredicts** |
| `ex_no_retire.empty:u` | **−5.150%** | 0.357% | **14.4 se** | **−9.93 cycles** |
| `cycclean:u` | **+0.186%** | 0.137% | 1.4 se | **+7.47 cycles** |
| `cycles:u` | +0.106% | 0.161% | 0.7 se | +4.3 cycles |

**The mechanism is confirmed at 14 standard errors and then it is priced, and
the price is bad.** The arm really does remove the mispredicts study 34 aimed at
— 1.088 per node, 158/159 rounds, 44 se — and they really are worth front-end
bubbles: 9.93 cycles of them. But 9.93 cycles is what the 42.58 extra
instructions cost, and then some. The net came out positive at **every** panel
size I looked at: +0.333% (n = 66), +0.176% (n = 118), **+0.186% (n = 159)**.

### 3.3 Study 34's model, corrected by two measured coefficients

Study 34 priced the trade at ≈ **−0.17% of cycles** from `−0.96 mispredicts ×
~18 cycles` against `+41.8 instructions` issuing at roughly 4 per cycle. Both
coefficients are now measured on this arm, in this engine, and **both are worse
for the arm**:

| coefficient | study 34 assumed | **measured here** |
|---|---:|---:|
| cycles of front-end bubble per mispredict removed | ~18 | **9.13** (9.93 / 1.088) |
| marginal instructions per cycle for the added work | ~4 | **2.45** (42.58 / 17.40) |

The second falls out of the first: the net is +7.47 cycles/node and the benefit
is −9.93, so the added instructions cost **+17.40 cycles**. That is the shape of
the window loop — eight iterations carrying a loop-borne `q -= shift`
dependency, plus a store per lane — not eight independent ALU ops. And it is
corroborated independently by `outcopy40` in the *other* direction: it removes
16.01 instructions from a different hot function and gains 12.05 cycles, i.e.
1.33 instructions per cycle at that margin (§2.5, §5).

**Verdict: `insertwin34` stays OFF.** ≈ **+0.19% of cycles**, which at ARCHON's
own calibration constant (~120 Elo per doubling of speed, `exp-0154`) is about
**−0.3 Elo**. Small, and the direct net is only 1.4 se on its own — but the
mechanism side is measured at 14 se, the instruction cost is exact, and the
model closes on both. **For the arm to break even the added instructions would
have to sustain 42.58 / 9.93 = 4.29 per cycle**, and nothing in this engine's
margins is near that.

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

**One strength change, and it came out of arm B rather than out of either
commissioned arm.**

`copy_acc_outlined` — the threat channel's pure-copy fold now goes through an
`#[inline(never)]` helper instead of expanding 40 zmm moves inline into
`at_full`'s fold loop. **−16.01 instructions and −1.21 branches per node**
(−0.270% / −0.195%), 8 paired rounds with zero spread, mispredicts unmoved,
node-identical at 5,001,562 and eval-hash-identical at `2e1ede9521624dd8`. Its
cycles contrast against the byte-identical null is **−0.354% ± 0.172%**
(2.1 se) — the right sign, and consistent in size with the instructions
removed.

**Two switches decided, with their doc comments rewritten so neither gets
re-opened blind:**

* `insertwin34` — stays OFF, now with the number that decides it and the
  exchange rate to re-price it against if anyone widens the window.
* study 38's `~26 i/n` — closed; the `null38` feature's own doc no longer
  claims to price "the machinery" on its own.

**Five instruments, all off by default and none of them shipped:**

| feature / tool | what it is for |
|---|---|
| `--features noalias40` | the alias machinery deleted at ONE site (a zero-sized `AliasTable` whose `get(k)` is `k`) — study 38's base, reconstructed in place |
| `--features callcopy40` | the pure-copy fold pinned to the library call, so machinery can be measured with codegen held fixed |
| `--features dosesort40` (`src/dose40.rs`) | the dose amplifier that **failed**, kept with the reason in its doc (§4) |
| `tools/perf40/copycheck40.sh`, `symscan40.py`, `textmd5.sh` | read the two codegen properties this study depends on OFF THE BINARY |
| `tools/perf40/memcost40.c` | what one 2560-byte `memcpy` actually retires (156.0 instructions, 20.0 branches) |

`--features outcopy40` is **gone, because it landed**: the outlined copy is now
the only path, with `callcopy40` still able to override it for measurement.

**And the panel kit**: `panel40.sh` (shuffled order inside the round, optional
per-label dose), `med40.py` (sign count + exact binomial p), `spread40.py` (the
noise's *shape*, and how many rounds an effect needs), `contrast40.py` (the
arm-minus-null contrast with a standard error), `derive40.py` (the
SMT-corrected cycle counter), `cpubusy40.sh` (which core pair is quietest).

### 5.1 Reproducing

```bash
bash tools/perf40/buildall40.sh          # b40 (main), w40 (insertwin34), n38
bash tools/perf40/buildarms40.sh         # the arm-B ladder + the dose binaries
cp bin34/b40 bin34/z40                   # the byte-identical null
bash tools/perf34/ident34.sh b40 w40 n38 na40 nac40 n38c   # node + eval-hash gate

# the codegen fact that reprices arm B (Sec 2.1)
bash tools/perf40/copycheck40.sh b40 n38 na40

# arm B's ladder, exact counters (Sec 2.2, 2.3)
P40_PANEL=docs/research/perf-artifacts-40/panel-armb.txt \
  bash tools/perf40/panel40.sh 12 b40 z40 n38 na40 nac40 n38c
python3 tools/perf40/med40.py docs/research/perf-artifacts-40/panel-armb.txt b40 \
  --metric instructions:u

# the copy rates (Sec 2.3)
ARCHON_FOLDCOUNT=/tmp/fc.txt ./bin34/fc40  bench 16 1 100000 \
  tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn

# what one 2560-byte memcpy retires (Sec 2.4)
gcc -O2 -o /tmp/memcost40 tools/perf40/memcost40.c
perf stat -e instructions:u,branches:u /tmp/memcost40 3000000 2560

# arm A: the bubble panel (Sec 3)
P40_EV=instructions:u,cycles:u,ex_no_retire.empty:u,\
ex_no_retire.thread_not_selected:u,ex_ret_brn_misp:u \
P40_PANEL=docs/research/perf-artifacts-40/panel-frontend.txt \
  bash tools/perf40/panel40.sh 159 b40 z40 w40 oc40
python3 tools/perf40/derive40.py docs/research/perf-artifacts-40/panel-frontend.txt \
  cycclean:u=cycles:u-ex_no_retire.thread_not_selected:u > /tmp/fe.txt
python3 tools/perf40/contrast40.py /tmp/fe.txt b40 z40 w40 oc40 \
  --metric ex_no_retire.empty:u

# the amplifier that failed (Sec 4)
bash tools/perf40/identdose40.sh ds40 0 12 24
P40_PANEL=/tmp/dose.txt bash tools/perf40/panel40.sh 2 ds40:0 ds40:24 dsw40:0 dsw40:24
```

Raw panels: `docs/research/perf-artifacts-40/` — `panel-core.txt` (n = 60, the
four-label core), `panel-armb.txt` (the ladder), `panel-frontend.txt`
(n = 159, the bubble panel), `panel-cycles.txt` (n = 18, the abandoned
direct-cycles panel, kept because it is the run that showed 45 s/round and made
me change instrument), `panel-land.txt` (the landing-tree check),
`foldcount-{main,noalias40}.txt`, `gates.log`. `bin34/` is a build output and is
not tracked.

---

## 6. GATES

`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` — **"All checks
passed."**

| gate | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| feature-matrix `cargo check` | clean |
| `cargo test` (debug: unit + fast perft tier) | 612 passed, 0 failed |
| release perft suite (`--ignored perft_full_suite`) | pass |
| `cargo test --features sota,arch-d48x` | pass |
| bench perft fingerprint | deterministic, identical |
| `bench 16 1 6 default depth` | **587683** — the pin, bit-for-bit |
| wl50 eval hash | `2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4` — the pin |
| d13 node total | **1739588** — the pin |
| node identity, study frame | **5,001,562** on all 12 binaries in every panel |
| landing tree vs measured binary | `land40` retires **5907.956 i/n / 619.310 br/n**, identical to `oc40` to six digits |

**Clippy under the opt-in features.** `--features sota,arch-d48x` has **9
pre-existing** `-D warnings` errors, all in files this study did not touch
(`sota_loader.rs`, `sota_simd.rs`, `aligned.rs`, `protected_halfka.rs`,
`simple_halfka.rs`); the merge gate only runs clippy on default features, which
is why they are there. **None of this study's code produces one** under
`sota,arch-d48x`, `+dosesort40`, `+noalias40` or `+callcopy40` — checked
per-file, and one that did appear (`copy_acc_outlined` dead under `callcopy40`)
was fixed with a `#[cfg]`.

### 6.1 A process hazard, paid for here: **never `git stash` in a worktree**

`refs/stash` is **shared across every worktree of the repository**. A `git
stash` in a clean worktree stashes nothing and silently succeeds; the matching
`git stash pop` then pops **another worktree's** entry into your files. It
happened here (mid-gate, checking whether a clippy failure was pre-existing) and
it left `src/movepick.rs` in a conflicted state with someone else's work in it.

It has happened before: the surviving `stash@{0}` on this repository is
literally titled *"RESTORED by perf-study-24: popped by mistake from
worktree-agent-af9e0a5aab3c0063e (refs/stash is shared across worktrees)"*.
Recovery both times: the entry is **kept, not dropped**, on a conflicted pop, so
`git restore --source=HEAD --staged --worktree <file>` puts your tree back and
the other worktree's stash survives. **The safe way to ask "is this failure
pre-existing?" is to build the reference commit into its own `--target-dir`, not
to stash.**

---

## 7. WHAT I AM LEAST SURE ABOUT

In order, most doubtful first.

1. **Arm A's NET, as opposed to its mechanism.** The benefit side is settled —
   −5.087% of `ex_no_retire.empty` at **12.2 standard errors**, 97/117 rounds.
   The net is not: the `cycclean` contrast reads **+0.333% ± 0.191% at n = 66
   and +0.176% ± 0.159% at n = 118**, i.e. between 1.1 and 1.7 se, positive at
   every n I looked at but never on its own conclusive. The verdict "it loses"
   rests on the *conjunction* of a measured −9.8-cycle benefit, an exact +42.58
   instruction cost, and a net that is positive in three independent readings.
   If someone re-runs this on an idle box and gets a negative net, the number to
   attack is the marginal instruction cost, not the bubble saving.
2. **Whether `ex_no_retire.empty` captures the WHOLE mispredict penalty.** It
   counts cycles in which nothing retired because the front end delivered
   nothing, which is where a mispredict's refetch bubble lands — but recovery
   work that overlaps with other retirement is invisible to it, so **8.71 cycles
   per mispredict is a lower bound on the benefit** and the arm's true net could
   be better than measured. I did not run the full `ex_no_retire.{all,
   not_complete, other}` decomposition that would bound the other side; the
   engine is 72% `ex_no_retire.all` and 55% `not_complete` (a single
   multiplexed sample), so there is a lot of room in there for me to be wrong
   about where a mispredict's cost lands.
3. ~~**The copy RATES are inherited.**~~ **Retracted before publication** — I
   wrote this caveat, then re-ran `foldcount` on both configurations of this
   tree and got 1,694,891 / 109,963 / 792,464 exactly (§2.3). The per-copy
   figures (111.83, 41.24, 5.57) are now measured end to end. What survives of
   the worry: those are *averages over a mixed population* — the 6.5% of
   pure-copy folds that survive the alias are the ones where only ONE
   perspective went empty, and there is no reason their per-copy cost should
   equal the both-empty case's. The decomposition in §2.3 does not depend on it
   (it uses only the i/n deltas); the "111.8 per copy" framing does.
4. **`#[inline(never)]` is a compiler-version-dependent win.** The −16.01 i/n is
   real on this toolchain and this build configuration, and the whole point of
   §2.1 is that LLVM re-takes this class of decision per context. What makes it
   safer than study 14's comment is that it is now *checkable*
   (`tools/perf40/copycheck40.sh` reads it off the binary) rather than asserted;
   what makes it unsafe is that nothing runs that check automatically.
5. **None of this has been arbitrated by games.** −16.01 i/n is ~0.27% of
   instructions, worth well under an Elo at ARCHON's own calibration constant,
   which is far inside the noise of any match this project can afford. It is
   landed on exact counters, which is the same standard studies 34, 37 and 38
   landed theirs on — but by the standing rule it has not been arbitrated.
6. **The `dosesort40` failure is diagnosed but not fixed.** I am confident the
   predictor learns the replay (the mispredict amplification is 1.6× where the
   design predicted 25×), and I am *not* confident that a bigger, sparser ring
   would fix it — I did not build one, because the amplifier has a second,
   independent fidelity failure (§4, last paragraph) that a bigger ring would
   not touch.
