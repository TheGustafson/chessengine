# Perf study 68 — the frontend wall attacked as LAYOUT (it is not layout), and the second form of the `qtrace` defect

**Date:** 2026-08-01 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, **32 KB
L1i**, 32 KB L1d, 1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at
**`e9c99aca`** (study 67 merged) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Instruments:** `tools/perf68/` (+ study 67's
`tools/perf67/`, reused, not rebuilt) · **Artifacts:**
`docs/research/perf-artifacts-68/` (four panels, 76 rounds).

**Base note (see §7.0):** `main` moved to `09968a46` mid-study and reverted the
SF tunable refresh; every panel here is on `e9c99aca` binaries, and the merge
gate is green on the merged tree at main's pins.

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,560 nodes**, one thread, 16 MB TT, pinned to
core 2. Every binary is PGO (`tools/build_pgo.sh sota,arch-d48x <name>
engines/canonical/net.cenn`, eval-file mandatory). A GPU trainer and a
continuous match shared the box throughout; **no `cycles:u` is quoted as a
verdict**, every speed claim is a within-round paired median with its sign count
against a **byte-identical binary null**.

---

## ★ 0. WHAT LANDED, AND THE THREE ANSWERS

### 0.1 Landed and gate-green

`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` → exit 0 on the
merged tree, `Bench: 587683`, eval hash `2e1ede95…79c4` — **every pin unmoved**
(§7). Node total on all seven PGO binaries in this study: **5,001,560, exact.**

1. **`SearchStats` is a zero-sized type in the shipped build** (§3). Eleven
   counters, fifteen per-node/per-move read-modify-writes, one reader — an
   `if std::env::var_os("ARCHON_SEARCH_STATS")` evaluated at **run time**.
   **−0.545 % instructions (15/15) = −29.1 per node**, node count and eval hash
   exact.
2. **The process-constant env probes are snapshotted per `go`, not re-read per
   node** (§4). `ord_weight()` is a `LazyLock<f32>` and it is the *first* test at
   the P-ORD site. **A further −0.014 % (16/16) = −0.75 per node — real, 16/16,
   and an order of magnitude less than I priced it at; §4.3 says why, because
   the mis-pricing is the transferable part.** Total landed: **−0.559 %
   instructions, −29.9 per node**, reproduced in **three independent panels**
   (15/15, 16/16, 14/14) to three decimal places.
3. **`tools/perf68/`** — six scripts, one of which is a new instrument:
   `statics68.sh`, a **named-static reference census inside a hot symbol**. It is
   what found §4 and what prices §6's next target.
4. **`PGO_EXTRA_RUSTFLAGS`** in `tools/build_pgo.sh` — codegen flags appended to
   the optimised rebuild only; unset reproduces the previous pipeline
   **byte-for-byte** (verified, §2).
5. **`--features pad68`** (`src/lib.rs`, default OFF) — the study's **layout
   null**, and the thing that turned §2 from a wrong conclusion into a right one
   (§2.4).
6. **`search-stats` is in `tools/check.sh`'s feature matrix**, so the enabled
   shape of §3's counters cannot rot.

### 0.2 The three one-line answers

> **The frontend wall is NOT a layout problem, and this is now measured rather
> than argued.** LLVM's **machine function splitter** — the production-grade
> version of study 67's arm A, driven by PGO block counts instead of source
> intuition — compacted the hot region's address span **30.7 %** (243,920 →
> 169,136 bytes), cut `search::<NonPv>`'s hot body **24.4 %** and
> `iterative_deepening`'s **47.7 %**, and cost **+0.001 % instructions (0/14)**.
> Nothing moved. **A free, perfectly-executed layout arm converted nothing.** §2.

> **And the instrument the whole route was to be judged on does not survive its
> own null.** In one 14-round panel, a **byte-identical** binary "moved"
> `ic_cache_fill_l2` **+2.290 %** while a binary whose `.text` genuinely differs
> (`pad68`, one cold never-called function) moved it **+0.055 %**. So the
> counter's wander is the BOX, not the layout — which kills my own
> layout-lottery hypothesis and puts study 67's +3.2 % and −1.5 % fill claims
> inside the noise. Their instruction costs, and therefore their verdicts,
> stand. §2.4.

> **The largest thing found: the `qtrace` defect has a SECOND FORM, and it was
> shipping in fifteen places.** Study 67 fixed *"the sink is `cfg`'d to `{}` and
> the ARGUMENT still does the work"*. This is *"the sink is a RUNTIME `if` and
> the STATE still does the work"*: eleven `pub` counters on `Worker`, bumped per
> node and per move, read only by an `env::var_os` test. `self` is a `&mut
> Worker` live across the recursive call, so every `+=` is a real load/add/store
> and no optimiser may delete it. Their doc comment said *"plain adds … —
> bench-neutral"*. §3.

---

## ★★★ 1. THE PRE-REGISTRATION (committed `22c98255`, before the first panel)

Study 67 left a budget and a rule: **445.7 cycles of empty op queue ÷ 128.0 L1i
line fills = at most 3.48 cycles (~4.7 instructions) per fill.** Price the arm
before building it; never price an instruction-cache arm in bytes.

**Arm 1, the machine function splitter (`-C llvm-args=-split-machine-functions`).
Pre-registered price: ZERO instructions per fill removed.** MFS is a
MachineFunctionPass that moves basic blocks LLVM's own PGO counters call cold
into a separate section. No call, no argument setup, no spill — a hot→cold edge
becomes a jump to a far address, and the profile says those edges are rarely
taken. Anything above ±0.15 % on `instructions:u` (null ±0.000 %) is an
unpredicted cost and must be reported as one.

**Pre-registered prediction on the deciding counter: `ic_cache_fill_l2` DOWN.**
This was the falsifiable part, and it was chosen because it discriminates:

> Study 67 §4.3 concluded *"the bytes it removes were never FETCHED — a line
> with zero executed ops never enters L1i"*. **If that is the whole story, MFS
> moves the fill counter by ~0**, because it removes the same kind of bytes. A
> drop could only come from the population that explanation does not cover:
> 64-byte lines carrying hot AND cold bytes together, fetched for the hot part.
> **So the sign of this arm decides between two mechanisms, and either answer is
> a result.**

**Pre-registered secondary: `de_src_op_disp.decoder` DOWN** (study 67 measured
40.5 % of ops arriving from the decoder and did not attack it; op-cache
utilisation is driven by density, so compaction should move it). New counter,
no prior null; its null is measured in the same panel.

**Pre-registered gates:** node total 5,001,560 and eval hash `2e1ede95…79c4`,
exact, on every binary.

---

## ★★★ 2. ARM 1 — THE MACHINE FUNCTION SPLITTER. FREE, PERFECT, AND INERT.

### 2.1 The null, first

`s68-b0n` is an independent full rerun of `tools/build_pgo.sh` on the same
source with `PGO_EXTRA_RUSTFLAGS` unset. **`cmp` clean against `s68-b0` —
byte-identical.** That establishes three things at once: study 67's PGO
determinism result still holds; the new env hook is inert when unset; and the
panel's null is a second name for the same bytes, the only null that cannot
smuggle in a codegen difference.

### 2.2 What it did to the binary — a textbook layout transform

101 cold-split symbols, **126,023 bytes** moved out of hot function bodies, and
the linker grouped every one of them into a single contiguous region
(`0x8d180`–`0xb8942`) far from the hot code.

| | `s68-b0` | `s68-mfs` | Δ |
|---|---:|---:|---:|
| `search::<NonPv>` hot body | 45,565 | **34,443** | **−24.4 %** |
| `search::<Pv>` | 20,326 | 11,931 | −41.3 % |
| `qsearch::<NonPv>` | 20,123 | 14,881 | −26.0 % |
| `iterative_deepening` | 26,239 | **13,710** | **−47.7 %** |
| `finny_refresh_into` | 18,799 | 10,778 | −42.7 % |
| `.text` total | 771,205 | 772,981 | +0.23 % |
| **hot-region address span** | **243,920 B** | **169,136 B** | **−30.7 %** |

The span row is the one that matters: in `s68-b0` the hot symbols run from
`search::<NonPv>` at `0x4e280` to `uci::run` at `0x89b50`; in `s68-mfs` from
`0x4ae30` to `0x742e0`, with all 101 cold fragments parked together at
`0x8d180`–`0xb8942`. **This is exactly what "get the executed lines contiguous"
means, done by the tool that does it best.**

### 2.3 The panel — 16 shuffled rounds, reference `s68-b0`

| counter | **`s68-mfs`** | sign | null (`s68-b0n`) | sign |
|---|---:|---:|---:|---:|
| **`instructions:u`** | **+0.001 %** | 0/16 | −0.000 % | 9/16 |
| **`ic_cache_fill_l2`** | **+1.103 %** | 5/16 | −0.491 % | 10/16 |
| **`de_src_op_disp.decoder`** | **+1.323 %** | 2/16 | +0.383 % | 5/16 |
| `de_op_queue_empty` | −0.147 % | 8/16 | −1.069 % | 11/16 |
| `task-clock` | −0.544 % | 8/16 | −2.062 % | 10/16 |

Node total 5,001,560 and eval hash `2e1ede95…79c4` — exact, both sides.

**The price prediction was exactly right and the direction prediction was
wrong.** +0.001 % instructions is 0.06 per node: MFS really is free. The fill
counter goes the wrong way rather than down, and so does the op cache.

At this point in the study I read the +1.103 % as a small real regression, wrote
a layout-lottery mechanism for it, and would have shipped that sentence. **§2.4
is what stopped me**, and it is the reason rule 1 says the null comes *first*.

### 2.4 ★★★ THE METRIC'S OWN NULL — AND `ic_cache_fill_l2` DOES NOT SURVIVE IT

Rule 1 says null the metric before reading the arm. MFS's fill number is
+1.103 %, and `ic_cache_fill_l2`'s byte-identical null across this study's
panels came back **−0.491 %**, **+0.431 %**, **+0.826 %** — a band the arm only
just clears. So I built the null the arm actually deserves.

`--features pad68` (`src/lib.rs`) adds **one cold, never-called function and
nothing else**: same nodes (5,001,560), same eval hash, **different `.text`**
(770,053 B — the pad displaced an inlining decision, which makes it a *more*
representative perturbation, not a less one). Its reference is `s68-c2v`, not
`s68-b0`, because it is built on the merged tree — a mistake I made and caught
when `s68-pad` read −0.559 % on `instructions:u` against the base, which is
exactly the landed delta and therefore exactly what an inert pad must NOT do.

**All five binaries, one set of 14 rounds**
(`docs/research/perf-artifacts-68/panel-layoutnull.csv`):

| counter | **`s68-b0n`** byte-identical null | **`s68-pad`** layout null *(vs `s68-c2v`)* | `s68-mfs` | `s68-c2v` *(landed)* |
|---|---:|---:|---:|---:|
| **`instructions:u`** | −0.000 % (7/14) | −0.000 % (9/14) | +0.001 % (0/14) | **−0.559 % (14/14)** |
| **`ic_cache_fill_l2`** | **+2.290 % (6/14)** | **+0.055 % (7/14)** | +1.790 % (5/14) | +1.928 % (4/14) |
| `de_op_queue_empty` | **+3.362 % (6/14)** | −0.489 % (8/14) | +0.666 % (7/14) | +0.281 % (7/14) |
| `de_src_op_disp.decoder` | +0.494 % (6/14) | +0.041 % (7/14) | +1.136 % (3/14) | +0.088 % (7/14) |
| `branch-misses:u` | +0.573 % (6/14) | −0.343 % (9/14) | −0.169 % (9/14) | +0.111 % (7/14) |
| `task-clock` | **+2.303 % (6/14)** | −1.382 % (8/14) | −0.981 % (8/14) | +0.966 % (7/14) |

> ### ★★★ Read the two null columns against each other
>
> **The binary that is byte-identical to the reference "moves"
> `ic_cache_fill_l2` by +2.290 %. The binary whose `.text` genuinely differs
> moves it by +0.055 %.**
>
> That pair kills two things at once:
>
> 1. **It kills my own layout-lottery hypothesis.** Two sections above I
>    proposed that shifting code re-rolls L1i set mappings and that this is why
>    the fill counter wanders. A real `.text` change moved it **0.055 %**. It
>    does not wander because of layout.
> 2. **It kills the counter as a verdict instrument at this resolution.** The
>    +2.290 % is the BOX — run-to-run state on a machine sharing a GPU trainer
>    and a continuous match — and it is **larger than every fill effect this
>    study or study 67 has ever reported.** `de_op_queue_empty` (+3.362 %) and
>    `task-clock` (+2.303 %) fail the same way in the same rounds.
>
> **Standing rule this earns:** `ic_cache_fill_l2`, `de_op_queue_empty` and
> `task-clock` may be quoted only **with a byte-identical null measured in the
> SAME panel**, never against a remembered one. Study 67 recorded a ~0.5 % fill
> null on its nights and then read a **+3.2 %** arm and a **−1.5 %** arm off it;
> at today's null neither of those clears. Their *instruction* numbers
> (+0.385 % and +1.517 %, both 0/14) are untouched and remain the reason both
> arms lost — **the conclusions stand, the fill evidence for them does not.**
>
> `instructions:u` is the only counter in the table that survives: −0.000 %,
> −0.000 %, +0.001 %, −0.559 %, with sign counts of 7/14, 9/14, 0/14, **14/14**.

### 2.5 ★ The mechanism, and what it does to study 67's arithmetic

> **MFS split out precisely the population study 67 measured as never
> executed.** `coldmap67.py` found **11,008 bytes** of `search::<NonPv>` in
> 64-byte lines that collect zero samples over the whole 5 M-node frame; MFS
> removed **11,122 bytes** from the hot body (and landed 11,249 in
> `search::<NonPv>.cold`, the difference being the jumps it had to add on the
> cold side). Those lines were never fetched, so relocating
> them cannot save a fill — and this arm proves it **without arm A's confound**:
> arm A paid +0.385 % instructions for its calls, MFS pays +0.001 %.
>
> **The measured statement, after §2.4, is a NULL and not a regression.** MFS's
> fill number (+1.103 % panel 1, +1.790 % panel 5) sits **inside** the
> byte-identical null measured in its own rounds (+2.290 %). The pre-registered
> prediction was **DOWN**; the measurement is **no movement detectable in either
> direction**, on every counter, at a resolution the instrument itself sets.
>
> **What is exactly measured is the price: +0.001 % instructions, 0/14.** MFS is
> free, it did everything a layout arm can do, and the frame did not notice.
>
> **And the address-span variable is dead.** Compacting the hot region 30.7 %
> for 0.06 instructions per node bought nothing. Span is not the quantity;
> **the number of distinct 64-byte lines the frame EXECUTES is**, and nothing
> that only relocates code can change that number.

**On the op cache**, which study 67 asked to be checked and nobody had measured:
`de_src_op_disp.decoder` reads +1.323 % (2/16) in panel 1 and +1.136 % (3/14) in
panel 5, against nulls of +0.383 % and +0.494 %. That is the one counter where
MFS's sign is repeatable and marginally outside its null, and it is a *mild
regression*, not the improvement predicted. I have no measured mechanism for it
and offer none. **What is settled: compaction does not improve op-cache
delivery.** The 40.5 % decoder share study 67 found is still unexplained and
still unattacked — by anything that works.

### 2.6 Verdict

**Not landed.** `s68-mfs` is not shipped and `PGO_EXTRA_RUSTFLAGS` stays empty
by default. The flag is left plumbed and documented so the arm is a one-liner to
re-run against a future tree, not a rebuild.

**Do not re-propose code layout, at any granularity, without a new mechanism.**
Study 67 closed source-level cold-path outlining and hot-callee deduplication —
on their *instruction* costs (+0.385 % and +1.517 %, both 0/14), which is the
half of its evidence §2.4 leaves standing. This closes **profile-driven
basic-block splitting with the production tool at literally zero instruction
cost**, which is the strongest form of the idea there is, and it closes it with
the counter that survives its own null.

---

## ★★★ 3. THE LANDED WIN — THE `qtrace` DEFECT'S SECOND FORM

### 3.1 The defect

`Worker` carried eleven `pub` counters — `qnodes`, `st_fh`, `st_fh_first`,
`st_fh_moves`, `st_ttcut`, `st_ext`, `st_research`, `st_lmr_n`, `st_lmr_plies`,
`st_nmp_try`, `st_nmp_cut` — bumped at **fifteen** sites, of which the hottest
are:

| site | rate |
|---|---|
| `self.qnodes += 1` (`search/mod.rs`, qsearch move loop) | **per qsearch node** |
| `self.st_ext += i64::from(extension)` | **per move**, main loop, unconditional |
| `self.st_lmr_n` + `self.st_lmr_plies` | 2 RMWs **per LMR-eligible move** |
| `self.st_fh` + `st_fh_first` + `st_fh_moves` | 3 RMWs **per beta cutoff** |
| `st_ttcut` ×2, `st_nmp_try`, `st_nmp_cut` | per TT-cutoff / per NMP node |

Their only reader is

```rust
if std::env::var_os("ARCHON_SEARCH_STATS").is_some() { eprintln!(...) }
```

**evaluated at run time.** Their doc comment claimed *"plain adds on paths that
already write stack state — bench-neutral, no decision reads them"* — the same
"believed free" claim `qtrace`'s doc comment made, and wrong for the same
reason. They are not adds: `self` is a `&mut Worker` live across the recursive
`search::<NT>` call, so each `+=` is a genuine **load / add / store to memory**
that cannot be kept in a register, and a runtime `if` cannot delete a store.

> ### The rule that covers both forms
>
> Study 67: *the sink is `cfg`'d to `{}` and the **argument** still does the
> work.* Study 68: *the sink is a runtime `if` and the **state** still does the
> work.*
>
> **A diagnostic that ships must be ZERO-SIZED when it is off, not merely
> unread. An empty body deletes a store only if there is no place to store.**

### 3.2 The fix

`SearchStats` is a struct with eleven fields under `cfg(any(test, feature =
"search-stats"))` and **a unit struct otherwise**; every bump is an
`#[inline(always)]` method whose body is a `#[cfg]`-gated block, so the shipped
build has neither the fields nor the adds. The methods take `Copy` scalars
already live in registers — the other half of the `qtrace` lesson, that an
argument must be free to construct. The report moves behind the same gate, which
matches the practice the research notes already recorded (*"archon-d47 has NO
SSTATS — use archon-sstats"*). `cfg(test)` implies the feature because two NMP
unit tests read `nmp_try`.

This is the idiomatic Rust shape for the problem: **the disabled configuration
is a type with no state, so "it compiles to nothing" is a property of the type,
not a hope about the optimiser.**

### 3.3 The panel — 16 shuffled rounds, reference `s68-b0`

`docs/research/perf-artifacts-68/panel-deadwork.csv`. Within-round paired
medians with the count of rounds in which the delta was negative.

| counter | base median | **`s68-c1`** | sign | null (`s68-b0n`) | sign |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | 26,713,785,780 | **−0.545 %** | **15/15** | −0.000 % | 9/16 |
| `branch-misses:u` | 111,846,341 | −1.424 % | 9/15 | +0.044 % | 8/16 |
| `ic_cache_fill_l2` | 633,963,434 | −0.045 % | 8/15 | +0.431 % | 6/16 |
| `de_src_op_disp.decoder` | 13,703,511,918 | +1.193 % | 5/15 | +0.181 % | 7/16 |
| `de_op_queue_empty` | 2,639,799,886 | −2.762 % | 8/15 | +0.152 % | 8/16 |
| `task-clock` | 4,385 ms | −1.551 % | 9/15 | −0.553 % | 8/16 |

**Node total 5,001,560 and eval hash `2e1ede95…79c4` — exact, on every binary
in this study.**

**The counter that decides it is `instructions:u`**: exact at fixed nodes, a
measured **±0.000 %** null, and layout-independent. **−0.545 % is −29.1
instructions per node**, at **15/15**. Against the frame's search work (total
minus study 67's measured 873 M-instruction startup ⇒ 5,166.5 i/node) that is
**−0.563 % of the search**.

The fill and op-queue columns are printed for completeness and **carry no
verdict** — §2.4 measures a **+2.290 %** byte-identical null on the first of
them, which swamps every number in that column. This change deletes ~15
read-modify-writes scattered across two 20–45 KB functions; it removes almost no
distinct *lines*, so a fill effect was never expected either way.

---

## ★★★ 4. THE THIRD FORM — THE GUARD DOES THE WORK

### 4.1 What `statics68.sh` found

`tools/perf68/statics68.sh` rolls up every RIP-relative reference to a **named
static** inside one hot symbol. On `s68-b0`'s `search::<NonPv>`:

| references | static |
|---:|---|
| 60 | `attacks::TABLES` (22 of them the `LazyLock` state word) |
| **8** | **`policy_head::ord_weight::W`** |
| **5** | **`search::nmp_allnode_probe::ON`** |
| **4** | **`policy_head::ord_min_depth::D`** |
| 12 / 11 / 9 / 8 / 8 | `psqt::PSQT`, `PAWN_ATTACKS`, `KNIGHT_ATTACKS`, `LINE`, `KING_ATTACKS` (real data) |

`ord_weight()` is a `LazyLock<f32>` over `CPL_ORD_W`, and it is the **first**
test at the P-ORD site, so the shipped build ran this at every interior
main-search node — quoted verbatim from `objdump` on `s68-b0`:

```
4f67f: mov  0xcac6b(%rip),%eax   # <policy_head::ord_weight::W+0x8>   <- Once state
4f685: test %eax,%eax
4f687: jne  ...                                                        <- init path
4f68d: vmovss 0xcac53(%rip),%xmm0 # <policy_head::ord_weight::W>       <- the f32
4f6a4: vpxor %xmm1,%xmm1,%xmm1
4f6a8: vucomiss %xmm1,%xmm0
4f6ac: jne  ... ; 4f6b2: jp ...
```

**Eight instructions, in the shipped binary, to re-learn that an environment
variable is unset.** A `LazyLock` deref reads an atomic, so no optimiser may
hoist it out of a recursive function. `nmp_allnode_probe()` is the same shape
(`OnceLock<bool>`) at every node reaching the null-move block.

*(Eight is the length of the sequence, not its cost per node. §4.3 measures the
cost and it is an order of magnitude smaller — that gap is the most transferable
thing in this section.)*

### 4.2 The fix, and why it is the codebase's own

Snapshot all three into `Worker` fields in `iterative_deepening`'s per-`go`
reset — the one entry both the pool path and the direct bench/test path share.
The values are process constants (env, read once through the `LazyLock`), so
behaviour is identical, and with `CPL_ORD_W` unset — every shipped
configuration — ordering stays bit-identical.

**This is perf study 13 §4.1's own fix, applied one level out.** That study
snapshotted `policy_head` itself for −4.8 i/n, and left the two knobs that
*guard* it being re-read per node: it moved the thing behind the door and left
the door's lock being picked at every node.

`statics68.sh` on `s68-c2` confirms the removal in the machine code: **all 17
env-probe references are gone** from `search::<NonPv>`; the `attacks::TABLES`
60 remain (§6).

### 4.3 The panel — and an honest downgrade

`s68-c2` is `s68-c1` **plus** the snapshot, so the incremental effect is the
`c1 → c2` contrast, measured in the same rounds:

| counter | **`s68-c2` vs `s68-b0`** | sign | **`s68-c2` vs `s68-c1`** | sign |
|---|---:|---:|---:|---:|
| **`instructions:u`** | **−0.559 %** | **15/15** | **−0.014 %** | **16/16** |
| `branch-misses:u` | +0.661 % | 4/15 | +2.159 % | 3/16 |
| `ic_cache_fill_l2` | +1.928 % | 6/15 | +0.828 % | 7/16 |
| `de_src_op_disp.decoder` | −0.166 % | 8/15 | −1.186 % | 10/16 |
| `task-clock` | −2.586 % | 9/15 | +0.814 % | 7/16 |

**The effect is real and it is much smaller than the disassembly suggested.**
16/16 on `instructions:u` is as consistent a sign as this panel produces, but
**−0.014 % is −0.75 instructions per node**, not the ~8 the machine-code
sequence in §4.1 implies. The reconciliation, stated rather than assumed:

- the P-ORD site is not per node — it sits after `MovePicker::new` in the main
  move loop, so only interior main-search nodes that reach it pay;
- **the snapshot is not free.** `Worker` is far too large to hold these in
  registers, so `mov (static); test; jne; vmovss (static)` becomes
  `vmovss self+off` — the branch and one load go, the `vpxor`/`vucomiss`/`jne`
  comparison stays. The saving is ~3 instructions per executed site, not 8.

**I priced this at ~8 i/node from the disassembly and it measured −0.75.**
Recorded here because the failure mode is instructive: *counting instructions in
a code sequence is not measuring them* — it silently assumes the sequence's
execution rate and ignores what replaces it. The instruction that is deleted and
the instruction that replaces it must both be counted, and only the panel does
that.

`branch-misses:u` at +2.159 % (3/16) is outside the ±0.04–0.6 % null it happened
to have in that panel and is **not explained**. It is quoted, not explained
away. (In panel 5's rounds the same binary reads +0.111 % on the same counter
against a +0.573 % null, so it does not reproduce; §2.4's lesson about
single-panel nulls applies to this counter too.)

**Landed** on the `instructions:u` verdict — 16/16, and the only counter in this
study that survives its own null in every panel.

---

## 5. MEASURED AND DECLINED — the 128-byte-per-node memset

`search::<NT>` opens its move loop with

```rust
let mut quiets_searched   = [Move::NULL; 32];
let mut captures_searched = [Move::NULL; 32];
```

and reads only `[..n_quiets]` / `[..n_captures]`, with `n` dynamic — so LLVM
cannot prove the tails dead. **The stores do ship**, verified in `s68-b0`:

```
4f6d1: vxorps %xmm0,%xmm0,%xmm0
4f6d5: vmovups %zmm0,0xdb0(%rsp)     <- 64 B
4f6e0: vmovups %zmm0,0x2b0(%rsp)     <- 64 B
```

Three instructions and 128 bytes of stack stores per interior node. The
codebase already owns the safe idiom that removes it (`MoveList::new`'s
`MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init()`, itself the fix
perf study 20 arm 4 landed for the identical defect).

**Declined on the §1 budget, not on difficulty.** Three instructions on the
~50 % of nodes that reach the move loop is ~1.5 i/node ≈ 0.03 %, against an
`unsafe` block and a new invariant to maintain. It is written down here with its
disassembly so the next session does not have to re-derive it, and so the
decision is a price rather than an omission.

---

## ★ 6. THE PRICED NEXT TARGET — `attacks::TABLES` IS A `LazyLock`

`search::<NonPv>` alone holds **60 references to `attacks::TABLES`, 22 of them
to the `LazyLock` state word.** Every `attacks(pt, sq, occ)` call anywhere in
movegen, SEE, check detection and `init_quiets` pays an atomic acquire load plus
a predicted branch before it can touch a magic table. `src/attacks.rs`'s own doc
comment already says so and calls it *"cheap"* — this study did not price it,
and nobody has.

It cannot become a `const` table: `AttackTables::new` searches for magics with a
randomised loop. The **safe** fix with no `unsafe` and no per-site discipline is
to give `Board` an `&'static AttackTables` field, set once in the constructors
(`empty` / `from_fen`) — once per board, not once per node — so every
`board.rs` / `movegen.rs` site reads a field instead of dereferencing a
`LazyLock`. That is a wider diff than this study could land and gate, and it is
the single largest remaining item of this class.

**Pre-price it before building it, per §1**: it removes instructions, so its
budget test is `instructions:u` (null ±0.000 %), and it should cost zero — an
8-byte field read replacing a static load plus a branch.

---

## 7. GATES

| gate | result |
|---|---|
| node total, **every** PGO binary in this study (`s68-b0/b0n/mfs/pad/c1/c2/c2v`) | **5,001,560** — exact |
| wl50 eval hash, every binary | **`2e1ede95…79c4`** — exact |
| PGO rebuild determinism (`s68-b0` vs `s68-b0n`) | **`cmp` clean, byte-identical** |
| `PGO_EXTRA_RUSTFLAGS` unset ⇒ previous pipeline | proved by the line above |
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` (default) | clean |
| `cargo check --all-targets` over the feature matrix, **`search-stats` now in it** | clean |
| `cargo test` (fast tier + checked perft mini-tier) | pass |
| release perft suite | pass |
| **`Bench:`** | **587683** — unchanged |
| **NNUE eval hash** (from source, non-PGO) | **`2e1ede95…79c4`** — unchanged |
| **NNUE d13 nodes** (from source, non-PGO) | unchanged |
| `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh`, **on the merged tree** | **exit 0** |

**No pin moves and none should.** §3 deletes state nothing reads; §4 reads a
process constant once instead of per node. Neither can move a node count, and
neither touches the forward pass.

### 7.0 ★ The base moved under the study, and the panels did not

`main` advanced from `e9c99aca` to `09968a46` while this study ran, and one of
those commits is **`d5f89245 Revert "refresh 154 search tunables from SF
eca43a97 → ebcea3ef"`** — the tunable refresh measured **−29.49 ± 14.88 Elo** and
was reverted. That moves `Bench:` back to **587683** and changes the shape of
the search tree, hence the frame's node count.

**Every panel in this document was run on binaries built from `e9c99aca`**, and
they are all internally consistent (same base, same null, same rounds, node
total 5,001,560 on all seven binaries). What the merge changes is the *tree*,
not the *work this study deleted*: fifteen read-modify-writes and three
`LazyLock` derefs are removed per node regardless of what the razoring margin
is. The merge gate above was re-run on the merged tree and is green against
main's pins.

**What is NOT claimed:** that −29.9 instructions/node reproduces to three
decimal places on the post-revert tree. It has not been re-panelled there. The
mechanism is tunable-independent, the magnitude was measured pre-revert, and
that distinction is stated rather than blurred.

### 7.1 ★ THE MEASURED BINARY WAS NOT THE LANDED BINARY — caught, and re-measured

Two edits landed **after** `s68-c2` was built and panelled:

1. `cargo fmt` collapsed §4's `if` onto one line (whitespace);
2. `SearchStats::default()` became `SearchStats::NEW`, because
   `clippy::default_constructed_unit_structs` fires on the shipped
   (unit-struct) shape and `check.sh` denies warnings — a lint that is a
   *warning* in a bare `cargo clippy` run and an *error* in the gate, which is
   why a clean manual clippy run did not catch it.

I wrote in this section that neither could change codegen, and then checked it
instead of asserting it. **It was wrong.** `s68-c2v` — a full PGO rebuild from
the merged tree — differs from `s68-c2` at **byte 25, the ELF `e_entry`**:
`.text` is 770,021 bytes against 769,973, a 48-byte difference somewhere in cold
code.

Every hot symbol is byte-for-byte the same size (`search::<NonPv>` 45,507,
`search::<Pv>` 19,910, `qsearch::<NonPv>` 20,123, `iterative_deepening` 25,481 —
identical in both), so the change is almost certainly confined to a cold path.
**"Almost certainly" is exactly the reasoning §2 and §4.3 both punish**, so the
whole panel was re-run against `s68-c2v`, the binary that actually landed:

| counter | base median | **`s68-c2v`** *(landed)* | sign | null (`s68-b0n`) | sign |
|---|---:|---:|---:|---:|---:|
| **`instructions:u`** | 26,713,785,698 | **−0.559 %** | **16/16** | −0.000 % | 9/16 |
| `branch-misses:u` | 110,676,402 | −0.097 % | 9/16 | +0.026 % | 8/16 |
| `ic_cache_fill_l2` | 613,522,756 | **+0.073 %** | 7/16 | +0.826 % | 7/16 |
| `de_src_op_disp.decoder` | 13,493,990,706 | −0.048 % | 9/16 | +0.234 % | 8/16 |
| `de_op_queue_empty` | 2,234,540,592 | +4.269 % | 5/16 | +1.894 % | 6/16 |
| `task-clock` | 4,022 ms | +1.209 % | 5/16 | +1.842 % | 6/16 |

`s68-c2v` reproduces `s68-c2`'s **−0.559 %** to three decimal places, at
**16/16**, and the two frontend counters come back **null** (+0.073 % and
−0.048 %, both inside a null that itself reads +0.826 % and +0.234 % today).
The re-measurement stands as the study's landed number; §3.3 and §4.3 remain the
attribution between the two changes, which needed `s68-c1` as a separate build.

> ### ★ And a near-miss worth recording
>
> At **n = 7** this same panel read `ic_cache_fill_l2` **+2.772 % (1/7)** and
> `de_op_queue_empty` +4.894 %, and I was one paragraph away from writing that
> the landed change carries a real instruction-cache regression and possibly
> reverting §4 on it. At n = 16 the fill number is **+0.073 %** and the null
> beside it is **+0.826 %**. **The counter was not regressing; the sample was
> too small, exactly as the standing rules say.** The only counter that read the
> same at n = 3, n = 7 and n = 16 was `instructions:u`, whose null is 0.000 %.

**The transferable rule:** *the binary you panelled is not the binary you
merged unless you `cmp` them.* A gate-driven lint fix between the panel and the
merge is enough to break it, and this one was invisible in every symbol size.

---

## 8. WHAT THE NEXT SESSION SHOULD DO

0. **Adopt §2.4's instrument rule before anything else.** `ic_cache_fill_l2`,
   `de_op_queue_empty` and `task-clock` get a byte-identical null **in the same
   panel** or they get no verdict. A remembered null is not a null on this box:
   the fill counter's byte-identical null read −0.491 %, +0.431 %, +0.826 % and
   **+2.290 %** on four consecutive panels of the same day. `instructions:u` is
   the only counter that held 0.000 % every time — build arms whose effect it
   can see.
1. **§6 — `attacks::TABLES`.** Priced, mechanism known, machine-code evidence in
   hand, and it is an INSTRUCTION-COUNT arm, which is the kind §2.4 says can
   still be judged.
2. **Re-run `tools/perf68/statics68.sh` after ANY new env knob or diagnostic
   lands**, on `search::<NonPv>`, `qsearch::<NonPv>` and `init_quiets`. Both
   §3 and §4 were invisible to every other instrument in this series and both
   were one command away. This class is the only one that passes study 67's
   budget for free, and it is the only class that has produced a win in two
   consecutive studies.
3. **Branch mispredicts remain completely unattacked.** The search side owns
   64.7 % of 21.72 misses/node; at ~15 cycles that is ~325 cycles/node, a
   separate ~8 % budget, and `init_quiets` alone is 15.89 % of it. Nothing in
   studies 67 or 68 touched it. Start by finding WHICH branch — the run-boundary
   test in `init_quiets`' scoring loop (`if !run.contains(m.from())`, taken at
   ~6 unpredictable positions out of ~23 quiets) is the first suspect and it is
   a suspect, not a finding.
4. **Do NOT re-propose:** code layout in any form (§2 — MFS is the strongest
   version of the idea and it is free and inert), cold-path outlining (study 67
   §4), hot-callee deduplication (study 67 §5), sub-byte FT weight formats or
   weight-row locality (studies 65/66), the history prefetch (study 44 §3), the
   TT-cutoff/eval reordering (study 64 §1.2).
