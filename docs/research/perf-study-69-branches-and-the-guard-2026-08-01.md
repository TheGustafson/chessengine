# Perf study 69 — the branch-mispredict profile the engine never had, and the third `qtrace` form swept exhaustively

**Date:** 2026-08-01 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1i,
32 KB L1d, 1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`e9501029`**
(study 68 merged + its handoff) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Instruments:** `tools/perf69/` (fifteen new;
`tools/perf67/` and `tools/perf68/` reused, not rebuilt) · **Artifacts:**
`docs/research/perf-artifacts-69/` (three panels, 38 rounds).

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes** on this base (study 68's frame was
5,001,560 — the tunable revert moved the tree), one thread, 16 MB TT, pinned to
core 2. Every binary is PGO (`tools/build_pgo.sh sota,arch-d48x <name>
engines/canonical/net.cenn`, eval-file mandatory). A GPU trainer and a
continuous match shared the box throughout; **no `cycles:u` is quoted as a
verdict** and every speed claim is a within-round paired median with its sign
count against a **byte-identical binary null measured in the same panel**.

---

## ★ 0. WHAT LANDED, AND THE THREE ANSWERS

### 0.1 Landed and gate-green

`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` → exit 0,
**`Bench: 587683`**, eval hash **`2e1ede95…79c4`**, d13 **1739588** — every pin
unmoved (§5).

1. **`tools/perf69/` — fifteen instruments, three of which this project did not
   have**: a **branch-mispredict profile down to the branch site, resolved to
   source** (`mispred69.sh` + `mispagg69.py` + `missrc69.py` + `misroll69.py`);
   an **execution-rate probe for never-taken code** (`opsat69.py`,
   `guardcost69.sh`); and a **whole-binary guard census ordered by measured cost
   per node** (`guardsweep69.sh`). Plus `buildnp69.sh`, the non-PGO control that
   turned §2 from a wrong conclusion into a right one.
2. **A measured negative, banner-committed at `attacks::tables()`** with both of
   its numbers, in the codebase's convention for a rejected arm.
3. **`--features pgonull69`** (`src/board.rs`, default OFF, in `check.sh`'s
   feature matrix) — the **PGO null**: eight inert bytes on `Board` in the exact
   place arm 1 put its pointer. It is what turned §2.5 from a wrong conclusion
   into a right one (§2.7), the same service `--features pad68` did for study
   68 §2.4.
4. **One counter added to the same-panel-null list**: `branch-misses:u`
   (byte-identical null **−0.803 %**, 11/16) joins `ic_cache_fill_l2`,
   `de_op_queue_empty` and `task-clock`.

**No strength change landed. The arm that was built lost, and the reason it lost
is the study's most transferable result.**

### 0.2 The three one-line answers

> **Study 68 §6's handoff — "`attacks::TABLES` is a `LazyLock`, it should cost
> zero to fix" — is measured and BOTH HALVES ARE WRONG.** The guard's whole
> budget is **4.98 dispatched ops per node**, not "60 references"; removing it
> is worth **−4.1 instructions/node** on the source (non-PGO pair, 10/10) and
> **+16.8** through the shipped PGO pipeline (**+0.321 %, 0/16**, against a
> byte-identical null of +0.000 %). Reverted. §2.

> **The larger finding: a source change that alters hot function SIZES buys a
> PGO inline re-decision, and it can cost 4× the change's own direct effect and
> invert its sign.** Arm 1 shrank five hot symbols by 108–903 bytes; PGO
> re-decided; 0.40 % of instructions moved in NNUE kernels the diff never
> touches. **This is causal, not a lottery — measured, not argued:** a new null
> (`--features pgonull69`, eight inert bytes on `Board`) that leaves every hot
> symbol's size *identical* reads **+0.000 % (5/12)** and reproduces PGO's
> decisions byte for byte. So the PGO number is the arm's true cost, the revert
> is right, and **when a PGO pair and a non-PGO pair disagree the gap is the
> inline re-decision and `symdiff69.sh` names the symbols responsible.** §2.5,
> §2.7 — including the retraction of §2.5's first draft, which had this
> backwards.

> **Branch mispredicts: the profile exists now, and the budget is 6.0 % of
> cycles spread over hundreds of sites with no concentration above 0.16 %.**
> At this engine's own measured **9.13 cycles per mispredict** and **2.45 IPC**
> on added scalar work (study 40 §3, not the generic 15 and 4), a branchless
> rewrite must cost under **22.4 instructions per mispredict removed** to break
> even; `insertwin34` cost 39.1 and lost by **1.75×**. The single largest source
> line in the engine —
> `movepick.rs:282`, the insertion sort's shift loop — is **4.49 %** of the
> taken-mispredict population, and **four distinct rewrites of it have lost**
> (studies 23, 34 and 55; re-priced by 40 and 56). §3.

---

## ★★★ 1. THE PRE-REGISTRATION (written and committed before the first panel)

### 1.1 The arm, and its price, measured before it was built

Study 68 §6 handed over `attacks::TABLES` as *"60 references inside
`search::<NonPv>` alone, 22 of them the `Once` state word"* and named the fix.
**A reference count is not a cost** — that is study 68 §4.3's own lesson, where a
sequence counted at ~8 instructions measured −0.75. So the guard was priced
first, by two independent instruments:

| instrument | what it answers | result |
|---|---|---|
| `tools/perf69/refoffs69.sh` | which BYTE of the static each reference touches | 22 refs to `TABLES+0x1040` (the `Once` state), 22 to `+0x0` and 16 to `+0x1000` (magic data) |
| `tools/perf69/guardcost69.sh` | how many times the guard EXECUTES per node, whole binary | **4.98 dispatched ops/node** across 15 symbols |
| `tools/perf69/guardsweep69.sh` | every guard-shaped static read in the binary | **4.21 ops/node total**, of which TABLES **3.40**, `THREAT_TABLES` 0.65, `CUCKOO` 0.15 |

Half the 22 sites in `search::<NonPv>` never execute at all (they are in cold
blocks); the two hottest execute 0.78 and 0.98 times per node.

**The sequence, quoted from `objdump` on the shipped PGO binary, is four
instructions and not three:**

```
mov  <TABLES+0x1040>(%rip),%eax      <- Once state, acquire
test %eax,%eax
jne  <init>                          <- never taken
lea  <TABLES>(%rip),%rdi             <- and this one is the surprise
```

The `lea` is structural, not incidental: **x86 RIP-relative addressing has no
index register**, so `magics[sq]` cannot fold the static's address into its own
operand the way a register base can. A `&'static` held in a register (or a
struct field) costs one `mov` and then indexes for free.

**Pre-registered price: −3 instructions per executed site.** At 3.40–4.98
sites/node that is **−10.2 to −14.9 instructions/node = −0.19 % to −0.28 %** of
the frame's 5,236.1 i/node.

**Pre-registered verdict counter: `instructions:u`, and only it.** It is exact
at fixed nodes, its byte-identical null has held 0.000 % across five panels of
studies 67–68, and this arm's effect is entirely an instruction-count effect.

**Pre-registered falsifications:**

- a reading of **0.000 %** falsifies the execution-rate measurement (the sites
  would not be running);
- a reading **worse than −0.10 %** means the field read is not being kept in a
  register across the uses the guard used to cover, i.e. the `lea` was
  double-counted;
- **any node-count or eval-hash movement at all is a bug**, not a result: the
  change is a representation change with no arithmetic in it.

### 1.2 The branch-mispredict target, and what it is allowed to conclude

The commission prices mispredicts at ~15 cycles. **This engine has measured its
own exchange rate and it is 9.13** (study 40 §3, from `ex_no_retire.empty`
against `ex_ret_brn_misp` over 159 paired rounds). Every cycle figure below uses
9.13, and the arithmetic is stated so it can be re-derived rather than believed.

Deliverable: a mispredict profile **down to the branch site**, which nothing in
this series could produce. §3.

---

## ★★★ 2. THE LANDED ARM — THE MAGIC TABLES ARE A FIELD, NOT A `LazyLock`

### 2.1 The defect, and the instruction the shape hides

`attacks::tables()` is `&*LazyLock<AttackTables>`. Every caller in movegen, SEE,
check detection and move scoring pays, in the shipped PGO binary:

```
mov  <TABLES+0x1040>(%rip),%eax     <- Once state, ACQUIRE
test %eax,%eax
jne  <init>                         <- never taken, ever
lea  <TABLES>(%rip),%rdi            <- and this is the one nobody counted
```

Three of those are the guard everybody expects. **The fourth is structural:
x86 RIP-relative addressing has no index register**, so `magics[sq]` cannot fold
the static's address into its own operand the way a register base can — the
compiler must materialise the base first. A `&'static` already in a register
indexes for free. That is why the arm removes four instructions and adds one,
not three-and-one.

An acquire atomic load cannot be hoisted out of a recursive function, so no
amount of `#[inline(always)]` deletes this: the codebase's own hot-loop rule
(*"call `tables()` once at function entry"*) hoists it out of a **loop**, and
PGO then inlines twenty such function entries into `search::<NonPv>`, where the
census finds **60 references, 22 of them the state word**.

### 2.2 The fix, and why it is the codebase's own idiom

`Board` gains an `AttackTablesRef` field — a `&'static AttackTables` newtype,
resolved **once in the constructor**. `Board::attack_tables()` replaces
`attacks::tables()` at every `board.rs` / `movegen.rs` / `movepick.rs` site.

The newtype is not decoration. `Board` derives `Debug, Clone, PartialEq, Eq`, so
a bare `&'static AttackTables` field would make derived equality **compare 4 KB
of magics on every `Board == Board`** and derived `Debug` print them. The
neighbouring `ThreatCache` already solves exactly this problem in exactly this
way — a private newtype whose `PartialEq` returns `true` and whose `Debug` is a
name — because it too is real state that is **not part of the position**. Adding
the second member of that family is a one-line-per-trait job and it makes the
"not the position" property structural instead of remembered.

*(The idiom, for the record: in Rust the way to keep a field out of `derive`d
semantics is a newtype with hand-written impls, not an attribute. There is no
`#[derive(PartialEq)]` skip; the type is the place to put the exception, and it
then applies to every derive on every container that ever holds it.)*

### 2.3 What it did to the machine code

| | `s69-b0` | `s69-c1` |
|---|---:|---:|
| `attacks::TABLES` references in `search::<NonPv>` | **60** (22 = `Once` state) | **0** |
| guard-shaped `TABLES` reads, whole binary (`guardsweep69.sh`) | 45 sites | **8 sites** |
| `TABLES` guard sites in `qsearch::<NonPv>` | 16 | **0** |

The eight survivors are the two NNUE threat paths and the startup force — the
sites that hold no `Board`. §5 prices them.

### 2.4 ★★★ THE PANEL — THE PRE-REGISTRATION IS FALSIFIED, AND IN THE OTHER DIRECTION

**16 shuffled within-round rounds, reference `s69-b0`, null `s69-b0n`
(an independent full rerun of `tools/build_pgo.sh` on the same source —
`cmp` clean against `s69-b0`, byte-identical).**
`docs/research/perf-artifacts-69/panel-guard.csv`.

| counter | base median | **`s69-c1`** (arm) | sign | null (`s69-b0n`) | sign |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | 26,186,645,500 | **+0.321 %** | **0/16** | **+0.000 %** | 6/16 |
| `branch-misses:u` | 110,032,390 | +0.133 % | 8/16 | **−0.803 %** | 11/16 |
| `ic_cache_fill_l2` | 610,061,494 | −1.217 % | 11/16 | +0.084 % | 8/16 |
| `de_op_queue_empty` | 2,218,850,826 | +0.171 % | 8/16 | −0.194 % | 8/16 |
| `de_src_op_disp.decoder` | 13,232,470,434 | −0.171 % | 9/16 | −0.108 % | 8/16 |
| `task-clock` | 4,001 ms | −0.194 % | 8/16 | +0.248 % | 7/16 |

Node total **5,001,562** and eval hash `2e1ede95…79c4` — exact on all three
binaries.

**Predicted −0.19 % to −0.28 %. Measured +0.321 %, 0/16 — a regression of
+16.8 instructions per node, at the most consistent sign this panel can
produce, against a null that read exactly 0.000 %.**

*(And an incidental result the commission asked for: **`branch-misses:u` does
not survive its own byte-identical null either** — the null reads −0.803 %
(11/16) in these very rounds. Any mispredict arm judged on it would need an
effect well above 0.8 % to mean anything. §3 uses LBR site histograms and the
PMU totals, never a paired `branch-misses:u` delta, for exactly this reason.)*

### 2.5 ★★★ THE MECHANISM — A NON-PGO CONTROL SEPARATES THE SOURCE FROM THE PIPELINE

The arm did exactly what it claimed (§2.3: 60 → 0 references in
`search::<NonPv>`), and every symbol it touches got **smaller**:

| symbol | `s69-b0` | `s69-c1` | Δ |
|---|---:|---:|---:|
| `qsearch::<NonPv>` | 19,897 | 18,994 | **−903** |
| `init_quiets` | 7,091 | 6,553 | **−538** |
| `init_captures` | 4,825 | 4,398 | −427 |
| `generate::<2>` | 4,937 | 4,696 | −241 |
| `see_ge` (306 → 273 instructions) | 1,333 | 1,140 | −193 |
| `search::<NonPv>` | 45,296 | 45,188 | −108 |

Smaller code and more instructions is a contradiction, so the arm was measured
a second way. **`tools/perf69/buildnp69.sh` builds the same two trees NON-PGO**
— identical inlining heuristics on both sides, no profile to re-collect, so the
pair measures the *source change* and nothing else
(`docs/research/perf-artifacts-69/panel-nopgo.csv`, 10 rounds):

| counter | base median | **`s69-c1-np`** | sign |
|---|---:|---:|---:|
| **`instructions:u`** | 26,922,468,344 | **−0.077 %** | **10/10** |

> ### ★★★ Two numbers for one diff, and both are true
>
> | pipeline | delta | per node | sign |
> |---|---:|---:|---:|
> | **non-PGO** — the source change alone | **−0.077 %** | **−4.1** | 10/10 |
> | **PGO** — the shipped pipeline | **+0.321 %** | **+16.8** | 0/16 |
>
> **Finding 1 — the guard is worth ~4 instructions per node, not ~15.** The
> field read *replaces* the `lea` rather than adding to it (both are one
> instruction; only one of them is a load), and the `Once` load and its branch
> were already partly common-subexpression-eliminated inside each function.
> This is **study 68 §4.3's lesson repeating exactly one study later**: it
> priced a `LazyLock` sequence at ~8 instructions from the disassembly and
> measured −0.75. I priced this one at 3 per site from the disassembly *and*
> measured its execution rate first, and was still 3.6× over — because a rate
> measurement prices the instructions REMOVED and says nothing about the ones
> that replace them.
>
> **Finding 2, and it is the larger one — a PGO-to-PGO delta is not a clean
> measurement of a source change.** The arm shrinks its callees; PGO's inline
> cost model re-decides; and **+0.40 % of instructions lands in symbols the
> diff does not touch.** Per-symbol `instructions:u` shares, `s69-b0` →
> `s69-c1`:
>
> | symbol | b0 | c1 | Δ share |
> |---|---:|---:|---:|
> | `SotaHead::forward_into` | 11.46 % | 12.17 % | **+0.71** |
> | `avx512::i8_deltas_into_inner` | 15.69 % | 14.88 % | **−0.81** |
> | `search::<NonPv>` | 30.18 % | 30.83 % | +0.65 |
> | `init_quiets` | 9.66 % | 9.17 % | −0.49 |
> | `see_ge` | 1.17 % | 0.97 % | −0.20 |
> | `finny_refresh_into` | 2.83 % | 3.08 % | +0.25 |
>
> The two NNUE kernels at the top are **not touched by this diff in any way**,
> and they move 0.7–0.8 share points in opposite directions. Converted to
> absolute counts (share × the run's own total, so the +0.321 % is divided out):
> `forward_into` **+6.5 %**, `i8_deltas_into_inner` **−4.9 %**,
> `search::<NonPv>` +2.5 %, `init_quiets` −4.8 %, `see_ge` −16.8 %. The
> search-side rows show the same thing from the other end: work migrated *into*
> `search::<NonPv>` out of `see_ge` and `init_quiets` — the inlining boundary
> moved.
>
> **Stated with its error bar, because a single `perf record` run is not a
> panel:** each profile is one run of ~13,000 samples, so a symbol at 11–16 %
> carries a ~2.2–2.6 % relative standard error and the two NNUE moves are
> **≈2σ each**, not 10σ. They are corroboration, not proof. The proof of the
> effect is the pair of panels — **+0.321 % (0/16) PGO against −0.077 % (10/10)
> non-PGO**, both of which are within-round paired medians with unanimous
> signs — and §2.7's null is what settles the *cause*.
>
> This is **study 55 §0.1's size curve seen from the other end**: that
> study measured `#[inline(always)]` buying −19.3 i/n on small bodies and
> costing **+25.7 i/n in spills** on a 3,516-byte one. Here the *callees*
> shrank, PGO pulled more of them into an already register-starved 45 KB
> function, and the spills followed.
>
> **The open question this leaves — is the re-decision CAUSAL or a lottery?**
> If it is a lottery, the arm's real value is the non-PGO −0.077 % and the
> rejection is an accident of one roll. If it is causal, the +0.321 % is the
> arm's true shipped cost and the rejection is correct. Nothing in studies
> 61–68 can tell these apart, so §2.7 built the null that can.

### 2.7 ★★★ THE NULL THAT SETTLES IT — AND IT CORRECTS §2.5's FIRST DRAFT

**`--features pgonull69`** (`src/board.rs`, default OFF, never ships): **eight
inert bytes on `Board`, in the exact place arm 1 put its `&'static
AttackTables`.** No code changes, no instruction the search executes differs by
intent — but every `Board` field offset after it moves, so instruction
encodings move (disp8 → disp32) and the binary is genuinely different.

Study 68's `--features pad68` cannot do this job: a never-called cold function
emits no profile counters, so the merged profile — and every inlining decision
— is identical. This one at least perturbs the struct the search hammers.

**What the binary says before the panel does:**

| | `s69-b0` | `s69-pgonull` |
|---|---:|---:|
| `search::<NonPv>` / `qsearch::<NonPv>` / `init_quiets` / `see_ge` / `init_captures` | 45,296 / 19,897 / 7,091 / 1,333 / 4,825 | **identical, all five** |
| `forward_into` / `i8_deltas_into_inner` | 3,694 / 33,131 | **identical** |
| `.text` | 769,701 | 769,797 (**+96**) |

**12 shuffled rounds, `docs/research/perf-artifacts-69/panel-pgonull.csv`:**

| counter | **`s69-pgonull`** | sign | null (`s69-b0n`) | sign |
|---|---:|---:|---:|---:|
| **`instructions:u`** | **+0.000 %** | 5/12 | −0.000 % | 9/12 |
| `branch-misses:u` | −0.162 % | 6/12 | +0.389 % | 6/12 |
| `ic_cache_fill_l2` | −0.161 % | 8/12 | −0.592 % | 9/12 |
| `de_op_queue_empty` | −1.041 % | 7/12 | −0.009 % | 6/12 |
| `task-clock` | −0.214 % | 6/12 | +0.046 % | 5/12 |

> ### ★★★ THE ANSWER — AND I HAD THE FRAMING WRONG
>
> **An inert perturbation that leaves every hot symbol's SIZE unchanged moves
> `instructions:u` by +0.000 %, and PGO makes byte-for-byte the same inlining
> decisions.** So:
>
> 1. **Growing `Board` by 8 bytes is innocent.** That was one of the two
>    candidate mechanisms for arm 1's regression and it is now eliminated by
>    measurement rather than by argument.
> 2. **The PGO re-decision is NOT the box, the roll, or `.text` layout, and
>    §2.5's first draft — which I wrote before this panel and which said "the
>    shipped-pipeline error bar on every arm in studies 61–68 is unknown" — is
>    RETRACTED.** The pipeline is *exactly* reproducible under a perturbation
>    that moves `.text` and every `Board` field offset but no function's size.
>    What provoked the re-decision was arm 1 **shrinking five hot symbols by
>    108–903 bytes**, and that is a property of the arm.
>    **What this null does NOT establish** (stated because the temptation to
>    over-read it is exactly what §2.5's first draft did): whether a size change
>    of a *given magnitude* has a predictable instruction cost, or a predictable
>    sign. It bounds the layout-only case at +0.000 % and localises the cause;
>    it does not calibrate it. §6 item 0 registers the follow-up.
> 3. **Therefore +0.321 % is arm 1's true shipped cost and the revert is
>    correct** — which is the opposite of the escape hatch the first draft was
>    building. The non-PGO −0.077 % is the arm's *direct* effect; the +0.40 %
>    difference is the *induced* effect, and both are the arm's.
>
> **The rule this earns, in its corrected form:**
>
> > **A source change that alters hot function SIZES buys a PGO inline
> > re-decision, and that re-decision can cost 4× the change's own direct
> > effect and can invert its sign. It is part of the arm and must be measured
> > on the PGO pair. A source change that does NOT alter sizes leaves the
> > pipeline exactly reproducible (+0.000 %).**
> >
> > Practical consequence: **when a PGO pair and a non-PGO pair disagree, the
> > gap is the inline re-decision, and it is diagnosable — `symdiff69.sh` names
> > the symbols whose size moved.** That is a lever, not a mystery: an arm
> > whose direct effect is a win and whose induced effect is a loss may be
> > recoverable by pinning the sizes it disturbed (`#[inline(never)]` on the
> > shrunk callees). Arm 1's direct effect is only −4.1 i/node, so it is not
> > worth that trouble; a bigger arm would be.

### 2.8 Verdict

**REVERTED** (`bd0103c2`). A measured +0.321 % on the shipped pipeline is not
shippable, whatever the non-PGO control says: the engine is built with PGO and
that is the number the engine runs at.

The result is preserved where it will be read — a banner on
`attacks::tables()` itself with both numbers, in the codebase's own convention
for a measured negative (`insertwin34`, `bcsort55`). Study 68 §6 called this
*"the single largest remaining item of this class"* and pre-priced it at
*"zero cost"*; **both halves are now measured and both are wrong**, and the next
session will read that at the definition rather than rediscover it.

---

## ★★★ 3. THE BRANCH-MISPREDICT PROFILE — DELIVERABLE 2

### 3.1 The instrument, and why nothing here could do this before

`perf record -e branch-misses` on AMD has **no PEBS-style precise mode**: the
sample IP is skidded and is useless at instruction resolution. That is why every
mispredict statement in this series has been per-SYMBOL.

This box is Zen 4 and has **AMD LBR v2**. `perf record -j any` captures a
16-deep last-branch stack at every sample, and each entry carries the branch's
own **source address**, its target, and a **mispredict flag** (`M/` vs `P/`). A
sample uniform in the instruction stream therefore yields a sample uniform in
the recent branch stream, and counting the `M`-flagged entries by source address
is an **unskidded per-branch-site mispredict histogram** with no skid to model
and no sample-IP to distrust.

The sampling event is `instructions:u` at a 2,000,003 period, two full frames.
**Not `cycles:u`** — studies 43/63 bar it on this shared box, and a
cycle-uniform sample would over-weight stalled regions, which is precisely the
population a mispredict profile must not be biased by.

### 3.2 ⚠ THE INSTRUMENT'S OWN LIMIT, MEASURED RATHER THAN ASSUMED

> **An LBR stack records only TAKEN branches.** A conditional branch that is
> never taken has **zero** entries no matter how often it executes.

This is not a footnote, it is the first thing the instrument taught, and it was
found the hard way: the ten `attacks::TABLES` guard branches located in
`search::<NonPv>` by `tools/perf69/guards69.sh` have **0 entries out of
418,976** between them, and they execute 2.20 times per node. `tools/perf69/
opsat69.py` (IBS dispatched ops) is the instrument for that population, and §2
is built on it.

The consequence for this section, stated in units:

| population | per node | in this histogram? |
|---|---:|---|
| `ex_ret_brn_misp:u` — all retired mispredicts | **21.65** | |
| `ex_ret_brn_tkn_misp:u` — mispredicts on branches that were TAKEN | **12.55** | **yes** |
| the difference — predicted taken, fell through | 9.10 | **no** |

**So this profile covers 58 % of the mispredict population and every share
below is a share of that 58 %.** The uncovered 42 % is not lost information —
it is per-symbol visible in the plain `branch-misses` profile — but it has no
site resolution here and no claim is made about it.

Sanity check on the scale: the LBR's own M-rate is 9.63 % of entries against a
measured `ex_ret_brn_tkn_misp / ex_ret_brn_tkn` of **7.22 %**. The 1.33×
over-read is a property of AMD's mispredict bit, is uniform across sites, and
does not affect a *ranking*; the absolute per-node figures below are scaled by
the PMU totals, not by the LBR rate.

### 3.3 The frame, in branches

| quantity | total | per node |
|---|---:|---:|
| `instructions:u` | 26,186,644,859 | **5,235.7** |
| `ex_ret_ops:u` | 26,937,674,601 | 5,385.9 |
| `branches:u` | 2,496,060,458 | 499.06 |
| `ex_ret_brn_tkn:u` | 869,661,811 | 173.88 |
| `branch-misses:u` = `ex_ret_brn_misp:u` | 108,263,499 | **21.65** |
| `ex_ret_brn_tkn_misp:u` | 62,756,953 | 12.55 |

**4.34 % of all retired branches mispredict — one in every 23.** 34.8 % of
branches are taken, and 58 % of the mispredicts land on those.

### 3.4 By symbol (`instructions:u`-sampled LBR, 418,976 entries, 2 frames)

| %of taken misp | rate | symbol |
|---:|---:|---|
| **37.31 %** | 11.8 % | `search::<NonPv>` |
| **11.21 %** | 10.5 % | `MovePicker::init_quiets` |
| 8.47 % | 8.8 % | `ThreatContext::at_full` |
| 6.02 % | 15.9 % | `sota_threats::record_square_touch` |
| 5.80 % | 4.5 % | `SotaHead::forward_into` |
| 4.82 % | 15.4 % | `HalfkaContext::finny_refresh_into` |
| 4.81 % | 5.5 % | `avx512::i8_deltas_into_inner` |
| 4.68 % | 8.0 % | `HalfkaContext::current_full` |
| 3.03 % | 15.8 % | `MovePicker::init_captures` |
| 2.84 % | 12.2 % | `qsearch::<NonPv>` |
| 1.48 % / 1.42 % / 1.17 % | | `search::<Pv>` / `init_evasions` / `see_ge` |

**Search side ≈ 60 %, NNUE side ≈ 40 %** — study 67's 64.7 % reproduces on a
different instrument and a different partition. `search::<NonPv>`'s 37.05 %
reproduces at **37.31 %**; `init_quiets`' 15.89 % reads **11.21 %** here (the
tunable revert changed the tree between the two measurements).

### 3.5 ★ BY BRANCH SITE, RESOLVED TO SOURCE — the thing nobody had

`tools/perf69/misroll69.py`, innermost `file:line` via `addr2line -i` against
the line-tables PGO twin (`s69-dbg`; same symbol sizes byte for byte, study 67
§1.1's proof re-checked here: `search::<NonPv>` 0xb0f0 and `init_quiets` 0x1bb3
in both).

| misp/node | %of taken misp | exec/node | rate | source |
|---:|---:|---:|---:|---|
| **0.5639** | **4.49 %** | 3.86 | 19.5 % | `movepick.rs:282` ← `:131` — the insertion sort's **shift loop** |
| 0.4576 | 3.65 % | 5.28 | 11.6 % | `sota_simd.rs:2614` |
| **0.4050** | **3.23 %** | 3.46 | 15.6 % | `movepick.rs:822` ← `mod.rs:3541` — `match self.stage`, the **movepick stage machine** (an indirect branch: 5–7 distinct targets observed) |
| 0.3593 | 2.86 % | 2.43 | 19.8 % | `sota_threats.rs:1995` |
| 0.2787 | 2.22 % | 0.79 | **47.2 %** | `movegen.rs:523` ← `mod.rs:3560` — `board.legal(m)`'s king-step arm |
| 0.2501 | 1.99 % | 0.72 | **46.2 %** | `tt.rs:538` ← `mod.rs:2623` — the TT cluster **tag scan** |
| 0.2457 | 1.96 % | 0.99 | 33.2 % | `macros.rs:180` ← `sota_threats.rs:1959` |
| 0.2280 | 1.82 % | 1.60 | 19.0 % | `movepick.rs:148` — the sort's `if tmp.value < limit { continue }` |
| 0.2168 | 1.73 % | 0.77 | 37.4 % | `sota_threats.rs:323` |
| 0.1901 | 1.51 % | 0.54 | 47.3 % | `non_null.rs:1714` ← `macros.rs:180` |
| 0.1869 | 1.49 % | 0.67 | 37.3 % | `range.rs:900` ← `:985` — the sort's outer `for p in 1..len` |
| **0.1652** | **1.32 %** | 0.47 | 47.0 % | `movepick.rs:1347` — **the run-boundary test**, study 54's `if !run.contains(m.from())` |
| 0.1493 | 1.19 % | 0.64 | 31.2 % | `mod.rs:3808` |
| 0.1014 | 0.81 % | 0.18 | **76.7 %** | `board.rs:1379` ← `mod.rs:3613` — `gives_check_by`'s direct-check test |

Full listing: `docs/research/perf-artifacts-69/mispred-b0-sites.txt` and
`-lines.txt`.

**The run-boundary test named in the handoff as "the first suspect" is real and
it is twelfth**, at 1.32 % of the taken-mispredict population.

### 3.6 ★★★ WHAT THE PROFILE SAYS, IN CYCLES, AND WHY NO ARM WAS BUILT ON IT

The commission prices a mispredict at ~15 cycles. **This engine measured its own
rate at 9.13** (study 40 §3: 159 paired rounds, `ex_no_retire.empty` against
`ex_ret_brn_misp`, −5.150 % of 192.91 cycles/node against −4.517 % of the
mispredicts). Using the engine's own number rather than the generic one:

```
21.65 mispredicts/node  ×  9.13 cycles  =  197.7 cycles/node
                                        =  6.02 % of cycles   (3,283 cycles/node at IPC 1.595)
```

**The budget is real and it is 6.0 %, not 8 %.** What the site profile adds is
the shape of it, and the shape is the finding:

> **It is not a budget with a hot spot. The largest single SOURCE LINE in the
> engine is 4.49 % of it — 0.564 mispredicts per node, 5.15 cycles per node,
> 0.157 % of cycles. The top FOURTEEN lines together are 26 % of it. The tail is
> flat for hundreds of sites.**

And the three largest concentrations have already been attacked, three times,
with the mechanism measured each time:

| target | share | prior attacks | measured result |
|---|---:|---|---|
| the insertion-sort shift loop (`movepick.rs:282`, `:148`, `range.rs:900` — **7.8 % together**) | 0.98 misp/node | study 23 §6 (`copy_within`, `partition_point`); study 34 arm 2 + study 40 §3 (`insertwin34`, branchless fixed-width window); study 55/56 (`bcsort55`) | **+91 i/n**, **+77.8 i/n**, **−1.088 misp/node for +42.58 i/n ⇒ +0.19 % cycles**, **+14.7 i/n** |
| the movepick stage machine (`movepick.rs:822`) | 0.41 misp/node | study 67 arm B (`icold67b`, deduplicate the four copies) | **+1.517 % instructions, 0/14** |
| the run-boundary test (`movepick.rs:1347`) | 0.17 misp/node | study 54 — it **created** this branch deliberately | **−21.3 i/n**; the mispredicts are the price it paid, and it is a good trade |

> ### The exchange rate is what closes this, and it is arithmetic, not opinion
>
> Study 40 measured, on this engine, that the instructions an arm adds to delete
> a mispredict retire at **2.45 IPC**, not the ~4 its designer assumed. So a
> branchless rewrite breaks even only if it costs fewer than
> **9.13 × 2.45 = 22.4 instructions per mispredict removed.**
>
> Apply that to the largest line available. Deleting **every** mispredict in the
> insertion sort — all 0.98 per node, which no rewrite has come close to —
> earns 8.9 cycles/node and may cost at most **22 instructions per node**.
> `insertwin34` removed 1.088 and cost 42.58. **It was not a bad
> implementation — at 42.58/1.088 = 39.1 instructions per mispredict removed it
> was 1.75× over a budget that is 22.4 instructions wide.**
>
> Every high-*rate* site in the table is worse, not better, because rate is not
> volume: `board.rs:1379` mispredicts **76.7 %** of the time and is worth
> **0.93 cycles per node**. A perfect fix there is 0.028 % of cycles.

**What was NOT done, and why it is a decision and not an omission.** Four
candidate arms were priced against that 22-instruction budget and all four are
outside it before a line is written:

1. **Branchless TT tag scan** (`tt.rs:538`, 0.250 misp/node): three `key16`
   compares → a 3-bit match mask + `trailing_zeros`. Earns 2.3 cycles/node.
   Roughly instruction-NEUTRAL, which means `instructions:u` — the only counter
   that survives its own null on this box — **cannot see it**, and no other
   counter here may issue a verdict (CLAUDE.md's perf-counter rule). An arm
   whose result is unmeasurable is not an arm.
2. **Branchless leaper tail in `attackers_to_exist`** (`movegen.rs:452`, 0.050
   misp/node): OR the pawn/knight/king terms instead of short-circuiting.
   0.46 cycles/node, and it *adds* the work of the terms the short-circuit
   skipped.
3. **Flattening `movepick.rs:822`'s jump table into a compare chain**: the
   dispatch is genuinely 5–7-way and the hot arms differ per node; every shape
   that avoids the indirect branch adds compares on the paths that do not take
   it. Study 67 already spent +1.517 % instructions on the neighbouring idea.
4. **`cmov`-ing the high-rate search decisions** (`mod.rs:2961` 65.9 %,
   `mod.rs:3188` 74.8 %, `mod.rs:3992` 56.6 %): each guards *expensive*
   work — a history update, a threat-map query, a reduction table walk — so
   removing the branch means computing the guarded body unconditionally. These
   are semantic decision points, and their unpredictability is the search
   working, not a defect.

**The honest statement, with its number: branch mispredicts are a 6.0 %-of-cycles
budget spread over hundreds of sites with no concentration above 0.16 % of
cycles, and this engine's own measured 22.4-instruction break-even makes every
branchless rewrite proposed so far a net loss.** That is a null with a
mechanism and an exchange rate, and the profile that produced it is reusable:
`tools/perf69/mispred69.sh` + `misroll69.py` regenerate the whole table in six
minutes against any future tree.

**What would change it** — stated so the next session has a live target rather
than a closed door: an arm that removes a mispredict *and* instructions at the
same time, i.e. one that deletes WORK rather than re-expressing a branch. §2's
guard removal is exactly that shape (it deletes a load, a test, a branch and a
`lea`), which is why it is the arm that got built — and §2.5 is why even that
shape is not enough on its own.

---

## ★★★ 4. THE DEAD-WORK SWEEP — THE THIRD FORM, DONE EXHAUSTIVELY

Studies 67 and 68 each found one instance of *"state or arguments that ship and
are discarded"* by **reading one symbol's disassembly**. That does not scale and
it does not terminate. `tools/perf69/guardsweep69.sh` does both: it walks the
whole `.text`, finds every `mov`/`cmp` of a **named static** whose value is
consumed by a conditional branch within three instructions — the machine shape
of a `LazyLock`/`OnceLock` deref or a process-constant re-test — and weights
each by IBS dispatched-op samples, so the output is **ordered by measured cost
per node rather than by reference count**.

`docs/research/perf-artifacts-69/guardsweep-b0.txt`, whole binary:

| ops/node | sites | guarded static |
|---:|---:|---|
| **3.4029** | 45 | `attacks::TABLES+0x1040` |
| **0.6518** | 5 | `sota_features::THREAT_TABLES+0x888` |
| **0.1534** | 2 | `cuckoo::CUCKOO+0x14000` |
| **4.2081** | 136 | **TOTAL** |

**That is the entire class, and it is 4.21 dispatched ops per node — 0.078 % of
the frame's 5,385.9.** Three statics; the other 133 candidate sites never
execute. §2 measured what removing the biggest one is actually worth (−4.1
i/node source-side, +16.8 shipped), so the two survivors are priced by the same
ruler at **−0.7 and −0.2 instructions per node** and are **declined on price,
not on difficulty** — the way study 68 §5 declined the 128-byte memset.

The other three forms were swept and are **clean**, which is the real finding
of this section:

| form | what was looked for | result |
|---|---|---|
| **"the argument does the work"** (study 67) | every function whose body is empty in the shipped `cfg`, and what its call sites construct | `qtrace` (Copy scalars since study 67); `MakeMoveHooks::{on_key,on_state}` (a `fold_rule50` xor and a `&`, both free and both DCE-able); `on_dirty` **is gated on `H::WANTS_DIRTY`** so `NoHooks` never builds the `DirtyPiece`; `AliasTable::set` is behind `noalias40`, not shipped. **Nothing to fix.** |
| **"the state does the work"** (study 68) | counters/history writes whose reader is disabled | `SearchStats` is a unit struct since study 68; `census34`, `ttcensus42`, `foldcount`, `skidprobe`, `dose40` are all `#[cfg(feature)]` and absent from `.text`; `tools/perf68/statics68.sh` on `search::<NonPv>`, `qsearch::<NonPv>` and `init_quiets` finds **no static that is not a real data table**. |
| **allocation on a discarded path** | `format!` / `to_string` / `to_owned` in `search/`, `movepick`, `movegen`, `board`, `tt`, `history`, and the four hot NNUE modules | **zero hits outside `cfg`-gated or once-per-`go` UCI code.** |
| **runtime `env` guards in hot paths** | `std::env::var*` reachable per node | all remaining ones are once-per-`go` (`Worker::new`'s `ARCHON_UNC_*` reads) or `search-control`-gated. Study 68 §4 took the last per-node ones. |

**The tunables are a null worth recording**, because they look like the defect
and are not: `QUIET_MAIN_HISTORY_WEIGHT()` and its ~260 siblings are function
calls in the hot loops, but `crate::tunables!` emits `#[inline(always)] pub fn
NAME() -> T { $default }` when `feature = "tune"` is off — a compile-time
constant, no atomic, no load. Only the `tune` build pays the `AtomicI64`.

> **The class is closed with a number.** After two studies of one-symbol reads,
> an exhaustive machine-code sweep of the shipped binary finds **4.21 ops per
> node** of guard and **nothing else at all** — and §2 shows the biggest single
> item of it is not worth taking. `guardsweep69.sh` should be re-run after any
> new env knob, `LazyLock`, or diagnostic lands; it is one command and it
> terminates.

---

## 5. GATES

| gate | result |
|---|---|
| node total, every PGO binary (`s69-b0/b0n/c1/dbg/pgonull`) and both non-PGO controls | **5,001,562** — exact |
| wl50 eval hash, every binary | **`2e1ede95…79c4`** — exact |
| PGO rebuild determinism (`s69-b0` vs `s69-b0n`) | **`cmp` clean, byte-identical** |
| line-tables twin (`s69-dbg`) is a valid address proxy | `search::<NonPv>` 0xb0f0 and `init_quiets` 0x1bb3 in **both** — study 67 §1.1's proof re-checked on this tree |
| `cargo fmt --check` / `cargo clippy --all-targets` | clean |
| `cargo test` (fast tier + checked perft mini-tier) | 614 pass |
| `cargo test --release -- --ignored perft_full_suite` | pass |
| **`Bench:`** | **587683** — unchanged |
| **NNUE eval hash / d13 nodes** (from source, non-PGO) | **`2e1ede95…79c4` / 1739588** — unchanged |
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` | **exit 0**, run on the arm tree AND on the reverted tree |
| merged tree vs panelled binary | the merged tree's `src/` differs from the panelled base by **a doc comment only** (`git diff --stat`: 1 file, 39 insertions, all `///`), so codegen is bit-identical by construction |

---

## 6. WHAT THE NEXT SESSION SHOULD DO

0. **★ PANEL EVERY ARM ON A PGO PAIR *AND* A NON-PGO PAIR, AND READ THE GAP.**
   §2.5/§2.7: arm 1's direct effect is −0.077 % (non-PGO, 10/10) and its shipped
   effect is +0.321 % (PGO, 0/16); the 0.40 % gap is the inline re-decision its
   own size changes provoked, and `--features pgonull69` proves that gap is
   causal (an inert perturbation that leaves symbol sizes alone reads
   **+0.000 %**). Both builds together cost ~10 minutes
   (`tools/perf69/buildnp69.sh` is the non-PGO half) and they turn "the arm
   lost" into "the arm's direct effect won and its induced effect lost, in
   *these* symbols" — which is a diagnosis with a lever
   (`tools/perf69/symdiff69.sh`, then pin the disturbed sizes) instead of a
   dead end. **The PGO number remains the verdict**; the non-PGO number is the
   diagnosis.
   *A further null still worth building*: one that perturbs the profile
   *counters* rather than the struct — e.g. an `#[inline(never)]` flip on a
   mid-sized callee — to see whether a size change of a given magnitude has a
   predictable instruction cost. §2.7's null deliberately does not move any hot
   symbol's size, so it bounds the "layout only" case and nothing more.
1. **Do NOT re-propose the `Board` attack-tables field** (§2, and the banner on
   `attacks::tables()`), the two remaining guards (§4, priced at −0.7 and −0.2
   i/node), or **any branchless rewrite of a mispredicting branch** without
   first checking it against §3.6's **22.4-instruction break-even**. Three
   independent studies have now lost that trade.
2. **The mispredict profile is reusable and takes six minutes**:
   `tools/perf69/mispred69.sh <bin> <out> 2 2000003` then `misroll69.py`.
   Re-derive it rather than quoting §3's table — the tunable revert already
   moved `init_quiets`' share from 15.89 % to 11.21 % between two studies one
   day apart.
3. **Remember which instrument answers which question.** LBR sees only TAKEN
   branches (58 % of the mispredict population) and is blind to a never-taken
   guard; IBS dispatched ops (`opsat69.py`, `guardcost69.sh`) sees every
   executed instruction and knows nothing about prediction. Using the first for
   §2 would have reported "the guard never runs".
4. **`branch-misses:u` joins the counters that need a same-panel null** — §2.4
   measured its byte-identical null at **−0.803 % (11/16)**, alongside
   `ic_cache_fill_l2`, `de_op_queue_empty` and `task-clock`. `instructions:u`
   remains the only counter on this box that held 0.000 %.

