# ARCHON perf study #38 — `thr_copy`: the price was a `memcpy` call nobody knew was there

**Date:** 2026-07-30 · **Base:** `main` at `859a442` (study 37 merged; canonical
net promoted to `network_scale = 600`) · **Instruments:** `tools/perf38/` on top
of studies 34/37 · **Artifacts:** `docs/research/perf-artifacts-38/`

---

## 0. HEADLINE

### 0.1 Study 37's `~26 i/n` was derived from code that does not exist

Study 37 §5 priced the threat channel's empty-delta copy at **~26 instructions
and ~26 load events per node**, from "40 zmm loads + 40 stores per copy" times
the 0.330 copies/node the foldcount census reports. It flagged the figure as
derived, not measured, and asked for it to be measured. It is now measured, and
**both the premise and the number are wrong.**

The premise: perf study 14 replaced `entry.acc[p] = parent.acc[p]` — which
lowered to a libc `memcpy` call — with a hand-chunked loop, and left this
comment in the source:

> `// Chunked so LLVM emits 40 inline 64-byte moves instead of one 2560-byte libc memcpy call.`

**On the shipped binary it does not.** `objdump` of
`ThreatContext::at_full` at `859a442`:

```asm
6f8e1:  mov    $0xa00,%edx                  ; 2560
6f8e6:  mov    %rbx,%rdi                    ; child entry
6f8e9:  mov    %r15,%rsi                    ; parent entry
6f8ec:  vzeroupper
6f8ef:  call   *0x7d2cb(%rip)               # ecbc0 <memcpy@GLIBC_2.14>
```

Twice — once per perspective. LLVM's loop-idiom pass recognises the chunked
loop straight back into a `memcpy`, so study 14's fix has been inert for some
unknown stretch of the project's history, and study 37 costed the source's
*intent* rather than the code that runs.

### 0.2 The measured price, by a doubled-copy probe

`--features probe-dblcopy` runs the same copy a **second** time into the ply
above (provably unread, node- and eval-hash-identical), so one paired panel
prices exactly one `memcpy` with the rest of the binary held fixed. Ten paired
within-round rounds, idle box, against a byte-identical null:

| counter | `dbl` (one extra copy) | null | per 2560-byte `memcpy` |
|---|---:|---:|---:|
| `instructions:u` | **+1.072%** [+1.07, +1.07] | −0.000% [0, 0] | **187.9** |
| `branches:u` | **+1.058%** [+1.06, +1.06] | −0.000% [0, 0] | **19.5** |
| `L1-dcache-loads:u` | **+2.185%** [+1.85, +2.53] | +0.082% [−0.22, +0.41] | **194.7** |

At the census rate of **0.338871 copies/node** (1,694,891 over 5,001,562 nodes)
the whole `thr_copy` site is therefore:

| | study 37, derived | **measured** | ratio |
|---|---:|---:|---:|
| instructions/node | ~26 (0.45%) | **63.7 (1.07%)** | **2.4× under-priced** |
| load events/node | ~26 (0.9%) | **66.0 (2.19%)** | **2.5× under-priced** |

**Study 37 under-priced this site 2.4×**, and it did so by costing 80 vector
moves where the machine executes a 188-instruction library call.

The branch figure is the check that the model is right: 19.5 branches per call
is ~10 loop back-edges plus ~9 entry-dispatch compares, which is exactly the
shape of the glibc kernel the profile lands in — a 4×64-byte
`vmovdqu64`/`vmovdqa64` loop at 11 instructions per 256 bytes, i.e. ten
iterations for 2560 bytes.

### 0.3 What landed, and what it did NOT get

**A ply whose resolved threat delta is empty for BOTH perspectives now points
at its parent's value holder and writes nothing at all.**

12–14 paired within-round rounds, idle box, node-identical at **5,001,562** and
eval-hash-identical at `2e1ede9521624dd8`:

| counter | reference/node | `alias` (landed) | null |
|---|---:|---:|---:|
| `instructions:u` | 5941.036 | **−0.270%** [−0.27, −0.27] | −0.000% [0.00, 0.00] |
| `branches:u` | 623.808 | **−0.433%** [−0.43, −0.43] | +0.000% [0.00, 0.00] |
| `L1-dcache-loads:u` | 3023.586 | **−1.653%** [−2.03, −1.45] | −0.085% [−0.26, +0.50] |
| `cycles:u` | 4199.258 | −0.495% [−5.86, +4.50] | **+0.771%** [−7.75, +8.57] |

**The load figure is the largest single-arm move on that counter in the 33–38
run**, five times its null's spread, and it is the counter study 33 §5.3
licensed as the cost proxy at these sites.

**And the honest half:** the arm removes 93.5% of the copies, so at 187.9
instructions each it *should* have delivered ≈ −61.8 i/n. It delivered
**−16.04**. **Something in the arm costs ~44 i/n**, and §5 reports what I could
and could not pin down about it. That is not a rounding error — it is 70% of
the available instruction win still sitting on the table, and naming it is more
useful than the part I banked.

---

## 1. METHOD, AND THE INSTRUMENT CORRECTIONS

### 1.1 The box, and a sixth null on `cycles`

All panels ran with the D49 trainer as the only other load. The commission
already retired `cycles:u`; this study adds the **sixth** independent null on a
**byte-identical** binary, and it is the largest yet on an otherwise idle box:

| counter | null median | null spread | verdict |
|---|---:|---|---|
| `instructions:u`/node | **−0.000%** | **[0.00, 0.00]** | exact — quotable |
| `branches:u`/node | **+0.000%** | **[0.00, 0.00]** | exact — quotable |
| `L1-dcache-loads:u`/node | −0.085% | [−0.26, +0.50] | ±0.3% floor |
| `ex_ret_brn_misp:u`/node | −0.522% | [−3.51, +1.68] | cannot adjudicate |
| `cycles:u`/node | **+0.771%** | **[−7.75, +8.57]** | **dead** |

No cycles claim is made anywhere in this document.

### 1.2 CORRECTION: `L1-dcache-loads` is not loads

Studies 33–37 (and §0 above, in their vocabulary) quote a counter called
"L1d loads". On this box that name resolves to `PERF_TYPE_HW_CACHE`
L1D/READ/ACCESS, which the AMD PMU driver maps to **PMCx040 `ls_dc_accesses`** —
*"the number of accesses to the data cache for **load and store** references"*.
Measured on one run of the study frame, the two are the same integer:

```
3345348843   L1-dcache-loads:u
3345348843   r0040:u                    <- identical
2534368961   ls_dispatch.ld_dispatch:u
 976317947   ls_dispatch.store_dispatch:u
```

So a "−1.653% L1d loads" result is **−1.653% of all data-cache accesses,
stores included** — which for a copy is the right thing to count (a 2560-byte
copy is 2560 B of reads *and* 2560 B of writes) but is not what the name says.
The landed arm removes ≈ 50 DC accesses/node, i.e. about **1.6 KB of L1 traffic
per node**. Prior studies' load numbers should be read with this in mind; none
of their *verdicts* change, because every one of them compared like with like.

### 1.3 Instruments

| tool | used for |
|---|---|
| `tools/perf34/build34.sh` | every labelled binary (MemoryMax=6G scope, `-j2`, ≥3000 MB gate) |
| `tools/perf34/ident34.sh` | identity gate: node count + wl50 eval hash |
| `tools/perf37/panel37.sh` + `med37.py` | the paired within-round panel |
| `tools/perf37/symcount37.sh` | static per-symbol instruction census |
| **`tools/perf38/memcpy38.py`** | libc `memcpy`/`memset` CALL-SITE census by symbol |
| **`tools/perf38/repmovsb38.sh`** | which glibc copy path a size takes (§4.2) |
| **`--features probe-dblcopy`** | the doubled-copy price probe (§0.2) |
| **`--features null38`** | the alias arm's own machinery, alias never taken (§5) |
| **`foldcount::THR_JOINT`** | the per-ply JOINT fold shape that chose the design (§2.1) |

---

## 2. THE ARM

### 2.1 Whole-entry, not per-perspective — and a census decided it

Study 37 named the blocker: `at_full` hands out `&[[i16; ACC]; 2]`, both
perspectives from one entry, so *a per-perspective alias cannot be returned
without changing the head merge kernel's parameter shape*. That is true. It is
also **avoidable**, if pure copies mostly come in pairs — which is a question
for a counter, not for reasoning.

`foldcount`'s new `thr_joint` histogram records, per folded ply, which
perspectives were needed and which resolved empty. On the study frame:

| cell | meaning | count | copies it causes |
|---|---|---:|---:|
| `n11e00` | both needed, neither empty | 2,095,420 | 0 |
| **`n11e11`** | **both needed, BOTH empty** | **792,464** | **1,584,928** |
| `n10e11` | only p0 needed, p0 empty | 55,487 | 55,487 |
| `n01e11` | only p1 needed, p1 empty | 54,476 | 54,476 |
| `n10e01`, `n01e10` | one needed, it folds | 82,494 | 0 |
| | **total** | **3,077,341** | **1,694,891** ✓ |

The total reproduces `thr_copy.calls` exactly. So:

* **93.5%** of all pure copies come from plies where both perspectives go empty
  **together**, and
* **25.75%** of all folded plies are pure copies of the whole entry.

The cheap form captures almost all of it and leaves `at_full`'s signature, the
head kernel and the SIMD backends untouched. The 6.5% remainder still copies.

### 2.2 The invariant

A per-ply `alias: Box<[u16; STACK_LEN]>` records **the ply whose entry holds
this ply's value**:

> if `fold[k][p] == Fold::Done` then perspective `p`'s value is in
> `stack[alias[k]].acc[p]`, and `alias[alias[k]] == alias[k]`.

The second half is why a read is always ONE hop: the alias case copies the
parent's *holder*, not the parent's index. Every write site maintains it —
`refresh`, the HalfKA-only early return, `discharge_refresh` and every
materialising fold set `alias[k] = k`; only the both-empty fold sets it to the
parent's holder. A ply marked `Refresh` for either perspective can never be
aliased, because the alias requires `FromParent` on both, which is what makes a
whole-entry alias sound in the mixed case.

The staleness argument is the module's existing one, extended one hop: every
ply on the way down is `apply_at`-pushed, which resets its `fold`, so a `Done`
ply is always one the current path materialised, and so is its holder.

### 2.3 Verification

* **Node-identical** (5,001,562) and **eval-hash-identical**
  (`2e1ede9521624dd8`) on every arm in every panel.
* **1,000,789 nodes of a debug build** with `assert_matches_refresh` live —
  the production oracle that compares `at_full(ply)` against a from-scratch
  refresh of both perspectives *and* both PSQT bucket sums at **every evaluated
  node**. Zero failures.
* `foldcount` on the arm: `thr_copy` 1,694,891 → **109,963**, `thr_alias`
  **792,464**, and `thr_into` unchanged at 4,270,334 — the real folds are
  untouched, exactly the 1,584,928 copies predicted are gone.
* `memcpy38.py`: `at_full` carries **2** `memcpy` references in `base` and
  **0** in the arm.

---

## 3. THE SAME SOURCE TEXT COMPILES BOTH WAYS

Worth recording on its own, because it is the reason §0.1 happened and it will
happen again.

| binary | the chunked copy compiles to |
|---|---|
| `base` | `call memcpy@GLIBC` (2 sites, `$0xa00`) |
| **`alias`** | a **fully unrolled inline 40×`vmovups` zmm copy** |
| `dbl` (probe) | `call memcpy@GLIBC` (4 sites, through a register) |

**Identical source text.** The chunked spelling is not a guarantee; it is a
decision LLVM re-takes per context and can withdraw. Study 14's comment
asserted an outcome the compiler never promised, and there was no gate that
could notice when it stopped being true.

`tools/perf38/memcpy38.py` is the gate that could: it reads the property off
the binary instead of off a comment. The in-tree comment now states the
mechanism and points at the tool rather than asserting the outcome.

---

## 4. TWO INERT ARMS, REPORTED IN FULL

### 4.1 The clamped-index arm — measured, ~null, NOT landed

The arm adds two indices LLVM cannot prove in range (`before[src]` and
`stack[alias[ply]]`), and `at_full`'s `panic_bounds_check` count goes 14 → 16.
The codebase's standard answer is a no-op range fact (`& 63`, `& (CAP-1)`), so
`alib` clamps both with `.min()`. Twelve paired rounds:

| label | `instructions:u` | `branches:u` | `L1-dcache-loads:u` |
|---|---:|---:|---:|
| `alias` (landed) | **−0.270%** | −0.433% | −1.653% |
| `alib` (clamped) | −0.247% | **−0.502%** | −1.591% |

It removes one of the two panic edges and trades ~0.023% of instructions for
~0.07% of branches. **Inside the practical floor and not obviously a win**, so
the landed arm keeps the unclamped form. More importantly it **refutes the
hypothesis** that the missing 44 i/n of §0.3 is bounds checking: a not-taken
bounds check is two instructions at ~0.6 executions/node, which cannot be 44.

### 4.2 The `rep movsb` hypothesis — WRONG, and my first probe was inert

I predicted the 2560-byte copy was a `rep movsb`, from the shape of the
numbers: a first pass through the alias delta implied ~50 instructions and
~158 load events per copy, and a ratio like that is what a microcoded string
instruction looks like (one retired instruction, many load ops). If true, it
would have meant `instructions:u` — the counter this run trusts most — was
nearly blind to the very site under study.

**It is wrong twice over.**

1. **The threshold.** `/lib64/ld-linux-x86-64.so.2 --list-tunables` reports
   `glibc.cpu.x86_rep_movsb_threshold: 0x1800000` — **24 MiB**, not the 2 KiB
   the folklore assumes. A 2560-byte copy was never near it. My first
   `repmovsb38.sh` run *raised* the threshold and measured a flat
   +0.000%/+0.000% — a null that was **inert by construction**, and the second
   run (lowering it to 512 B) measured +0.000% too, because
   **`GLIBC_TUNABLES` is not honoured on this box at all**:
   `GLIBC_TUNABLES=…=512 ld.so --list-tunables` still prints `0x1800000`. Two
   arms, no signal, no mechanism — recorded because an inert instrument that
   returns a plausible null is exactly how a study convinces itself of
   something false.
2. **The disassembly settles it anyway.** `perf record` puts the libc time at
   `0x189e40`–`0x189eaa`, which disassembles to a 4×64-byte
   `vmovdqu64`/`vmovdqa64` loop — the **AVX-512 `unaligned_erms` memmove**, 11
   instructions per 256 bytes. Not `rep movsb`.

And the numbers that made me suspect it were an artifact of dividing the ARM's
delta by the copies removed — i.e. of assuming the arm's only effect was
removing copies. §0.3 says it is not. The direct probe (§0.2) gives 187.9
instructions and 19.5 branches per call, both of which fit the disassembled
kernel to within a few percent.

---

## 5. THE MACHINERY COSTS 19.7 i/n, AND I PRICED IT RATHER THAN GUESSING

Arithmetic, from the direct probe:

```
  copies removed          0.316885/node x 187.9 i  =  -59.54 i/n
  residual copies, memcpy -> inline
                          0.021986/node x ~103 i   =   -2.26 i/n
                                                      ---------
  predicted                                           -61.80 i/n
  measured (`alias`)                                  -16.04 i/n
                                                      ---------
  overhead to account for                             +45.76 i/n
```

Rather than write a mechanism paragraph, I built the null. **`--features
null38`** carries the entire alias machinery — the array, its read in
`at_full`, its read of the parent's holder in the fold, its per-fold store —
with the alias **never taken**, so every copy still runs. Ten paired rounds:

| label | `instructions:u` | `branches:u` | `L1-dcache-loads:u` |
|---|---:|---:|---:|
| `anul` (machinery only, no copies removed) | **+0.332%** [+0.33, +0.33] | −0.554% [−0.56, −0.55] | +0.198% |

**+19.7 instructions per node for four array accesses per fold.** That is 43%
of the gap, measured rather than argued, and it is the single most useful
number in this study for whoever works here next: in this walk, *touching a
fifth per-ply array at all* costs ~20 i/n before it does anything.

Two caveats I will not paper over:

* The null is **imperfect**. `ALIAS_ENABLED` is a `const`, so LLVM deletes the
  `need[0] && need[1] && lens == (0,0)` test entirely; the null therefore
  carries slightly *less* than the arm does, and +19.7 is a lower bound.
* Its **branch** column cannot be composed with the arm's. The machinery alone
  measures −0.554% branches while removing no copies at all, which is a codegen
  artifact, not a saving — so the arm's −0.433% branches is not "the memcpy's
  branches removed" and I do not present it as one. The instruction and
  DC-access columns compose; the branch column does not.

That leaves ~26 i/n still unaccounted, and §4.3 shows the obvious structural
fix recovers almost none of it — so the residue is not the base pointer either.

### 5.1 What is left on the table

The site is worth **63.7 i/n** (§0.2) and the landed arm banks **17.1** of it
(§4.3). The remainder is not more copies to remove — it is the ~26 i/n this
study could not localise inside `compute`/`at_full`. A study that wants it
should start from `null38`, not from the arm.

---

## 6. WHAT LANDED

**`#[cfg]`-free, one change:** the both-empty threat fold aliases its parent's
holder instead of copying 2 × 2560 bytes.

* **−0.270% instructions, −0.433% branches, −1.653% data-cache accesses**, all
  against a null of −0.000% / +0.000% / −0.085%, 14 paired within-round rounds
  on an idle box.
* Node-identical and eval-hash-identical; 1.0 M nodes of debug-oracle search.
* In absolute terms: **−16.0 instructions, −2.7 branches and −50 DC accesses
  per node**, the last of which is ~1.6 KB/node of L1 traffic.

Three things go with it and are the durable part:

1. **`tools/perf38/memcpy38.py`** — the libc call-site census. Study 14's
   comment was a codegen claim in prose; this reads it off the binary.
2. **The corrected comment** at the copy, recording that the chunked spelling
   is a compiler *decision*, not a guarantee, with the three-way evidence of §3.
3. **`foldcount::THR_JOINT`** — the joint fold shape, which is what turned "a
   per-perspective alias needs the head kernel changed" from a blocker into a
   6.5% remainder.

---

## 7. GATES

| gate | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `cargo test` (debug: unit + fast perft tier) | pass |
| release perft suite (`--ignored perft_full_suite`) | pass |
| `cargo test --features sota,arch-d48x` | pass |
| `bench 16 1 6 default depth` | **587683** — the pin, bit-for-bit |
| wl50 eval hash | the `859a442` pin |
| d13 node total | the `859a442` pin |
| node identity, study frame | **5,001,562** on every binary in every panel |
| debug oracle | 1,000,789 nodes, zero failures |

**And the study-37 hygiene rule, kept:** `bin34/base` md5s to
`19798abf79732d92c9fbcb2c90af699f`, which is **byte-identical to study 37's
landed binary** — so this study's reference provably is `main`, and the kit
commit that precedes the arm is codegen-inert.

---

## 8. REPRODUCING

```bash
# the census that says the copy is a memcpy call
objdump -d bin34/base > /tmp/base.asm
python3 tools/perf38/memcpy38.py /tmp/base.asm | grep at_full

# the joint fold shape that chose whole-entry over per-perspective
bash tools/perf34/build34.sh foldc "sota,arch-d48x,foldcount"
ARCHON_FOLDCOUNT=/tmp/fc.txt ./bin34/foldc bench 16 1 100000 \
  tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep thr_joint /tmp/fc.txt

# the price probe (Sec 0.2)
git apply docs/research/perf-artifacts-38/dblcopy-probe.patch
bash tools/perf34/build34.sh dbl "sota,arch-d48x,probe-dblcopy"
P37_PANEL=docs/research/perf-artifacts-38/panel-price.txt \
  bash tools/perf37/panel37.sh 10 base null dbl
python3 tools/perf37/med37.py docs/research/perf-artifacts-38/panel-price.txt base

# the arm (Sec 0.3) and the clamped variant (Sec 4.1)
P37_PANEL=docs/research/perf-artifacts-38/panel-core.txt \
  bash tools/perf37/panel37.sh 14 base null alia
python3 tools/perf37/med37.py docs/research/perf-artifacts-38/panel-core.txt base

# the glibc path, and the tunable that is NOT honoured here (Sec 4.2)
/lib64/ld-linux-x86-64.so.2 --list-tunables | grep rep_movsb
bash tools/perf38/repmovsb38.sh 8 base
```

Raw panels: `panel-core.txt`, `panel-price.txt`, `panel-arms.txt`,
`panel-machinery.txt`, `panel-repmovsb.txt`, `panel-repmovsb-inert.txt`.

---

## 9. WHAT I AM LEAST SURE ABOUT

In order.

1. **That the 45.8 i/n gap is register pressure.** It is a residual, not a
   measurement: I priced the copy (§0.2, direct), priced the arm (§0.3,
   direct), and subtracted. §4.1 excludes bounds checks and the static census
   is consistent with spills, but "consistent with" is not "measured", and I
   have twice this session had a mechanism story fall over when I built the
   arm for it. `--features null38` exists precisely so the next session does
   not have to take my word for it.
2. **The 187.9 instructions per `memcpy`.** The panel is exact and its null is
   zero, so the *delta* is not in doubt; what is derived is dividing it by the
   copy rate, which assumes the probe's marginal copy costs the same as the
   original. Both are 2560 bytes, 64-byte-aligned both ends, non-overlapping,
   and land in the same glibc kernel, so I believe it — and the branch figure
   (19.5, against ~10 loop back-edges plus the entry dispatch) is an
   independent check that fits. But it is one probe.
3. **The DC-access granularity.** §1.2 pins *what* the counter is by an exact
   identity with `r0040`. What it counts per 512-bit access I did not
   establish: 194.7 accesses for a copy that issues ~40 zmm loads and ~40 zmm
   stores implies more than one event per access, and I did not chase it. The
   *relative* result is unaffected — every number here is a paired delta on one
   counter — but do not convert those percentages into bytes without settling
   it.
4. **How long study 14's fix has been inert.** I did not bisect. It is a
   memcpy call today and it was 40 inline moves when study 14 measured it; the
   commit that flipped it, and whether it was a rustc bump or a neighbouring
   edit, is unknown. §3 shows the same text going both ways in binaries built
   *today*, which makes "a specific commit broke it" the wrong question and
   "nothing was watching" the right one.
5. **Whether the residual 6.5% is worth anything.** 109,963 copies is
   0.022/node; at 187.9 they were ~4 i/n, and in the landed binary they already
   compile inline. A per-perspective alias would cost the head kernel's
   parameter shape for less than that. I did not build it and I do not think it
   should be built.
