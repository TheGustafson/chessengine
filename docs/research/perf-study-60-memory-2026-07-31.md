# Perf study 60 — the memory block, priced against what was already measured

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`c09e21be`** (post-study-59;
§3 was additionally measured on `450f766d` and re-measured after the rebase) ·
**Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf60/` · **Artifacts:**
`docs/research/perf-artifacts-60/` · **Branch:**
`worktree-agent-a835fa6f0b099db66`

**Rebase check (the `bcmisc55` rule).** The study opened on `450f766d`; `main`
merged perf study 59 (`sota_features.rs` + `sota_loader.rs`, −176.8 shipped i/n)
while it ran. Everything below §3 was rebuilt from source on `c09e21be` and
re-measured there; the pre-rebase panel is kept as
`panel-regs4.txt` and the post-rebase one as `panel-post59.txt`. The NNUE gate
pins are unchanged by study 59 (`2e1ede95…79c4`, d13 `1739588`), and all four
binaries reproduce them. `main` then moved again to **`2dde8ce2`** during
write-up; `git diff --stat c09e21be 2dde8ce2` is **one docs file**, so the
branch was rebased onto it and nothing was re-measured — the rule is that a
number expires when its *substrate* moves, and `src/` did not.

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT,
`-C target-cpu=native`, pinned to core 1. **Perf study 59 and a GPU training
run shared the box for the whole study**; §6 is what that did to the
instruments and it is the reason two of this study's questions are answered and
one is not.

---

## ★ 0. WHAT LANDED, AND THE ONE NUMBER THAT MATTERS MOST

**Landed: a hard guard in `tools/build_pgo.sh`, and it is worth more than any
arm in this document.**

`engines/canonical/net.cenn` is a symlink into `data/`, which is gitignored, so
**it dangles in every fresh git worktree** — which is where every perf study
since 40 has run. Before this guard, `build_pgo.sh` with a dangling eval-file
still ran its UCI training session (the `setoption` failed, the session searched
with PSQT), still merged a profile, still emitted a binary, and **said nothing**.
That is the failure its own header documents in capitals, with no check behind
it.

Measured here, median of 3, same frame, same source, non-PGO native as the
reference:

| build | `instructions:u` / node | vs non-PGO |
|---|---:|---:|
| non-PGO native (`base`) | 5819.68 | — |
| **PGO trained WITH the net** | **5728.98** | **−1.559 %** |
| **PGO trained WITHOUT the net** | **6040.90** | **+3.802 %** |

**A silently-untrained PGO binary is not "PGO inert" — it is 5.445 % worse on
instructions than the correctly-trained one and 3.801 % worse than no PGO at
all.** Both binaries are node-identical at 5,001,562, so that is a pure code
contrast; a three-run pair puts the cycles gap at **+0.885 %** (median of 3,
`tools/perf60/pair60.sh` — three runs is not a panel and is quoted as an
order of magnitude, not a result), and §5.1 is what the wall-clock panel said
about it. Every counter read off such a binary is a property of the profile, not
of the arm. Perf study 57 §10 escalated exactly this class of failure ("the PGO
baseline shows no PGO win at all… that conclusion needs its own check before it
becomes doctrine") and could not close it. The guard closes the mechanism: a
`[ -r ]` on the eval-file (which follows symlinks, so it is precisely the
dangling-symlink test) plus a grep for `ERROR EvalFile` in the training
session's own output, for a net that exists but does not parse.

**Honesty about what that does and does not prove.** Study 57 measured its PGO
baseline at **+0.060 %** instructions, not +3.802 %, so its anomaly is *not*
demonstrated to be this bug — the magnitudes disagree. What is established is
that the failure is real, silent, reachable from any worktree, and now
impossible.

**Also landed:** two off-by-default arms (`hskip60`, `htail60`) carrying their
own refutations, `tools/perf60/`, and the negative results in §3–§5.

**Not landed: any speed change. Three arms were built, gated node-identical and
eval-bit-exact, and all three lost:**

| arm | `instructions:u` | `L1-dcache-loads` | `L1-dcache-load-misses` (net of null) | verdict |
|---|---:|---:|---:|---|
| **`regs4`** — the REGS sweep's last untested tile width, and the only line-aligned one | **+1.583 % (+89.4/node)** | +0.78 % | **−0.07 %, 18/40 — a coin flip** | §3 — its mechanism is dead |
| **`hskip`** — MLP for the output-skip lane | **−0.031 % (−1.75/node), 40/40** | **−0.17 %** | **+0.48 %, p < 0.0001 — WORSE** | §4 — mechanism inverted |
| **`htail`** — prefetch the head's tail tables | +0.404 % (+22.8/node) | +0.25 % | **+0.24 %, p < 0.0001 — WORSE** | §5 — prefetch kill #9 |

Gate on the final tree: `CHECK_NNUE_GATE=1 bash tools/check.sh` →
**All checks passed.**, `Bench: 587683`, eval hash `2e1ede95…79c4`, d13
`1739588`; `cargo test --release --test perft_suite -- --ignored
perft_full_suite` → ok.

**§2 is why nothing landed, and it is the study's main result.**

---

## ★ 1. THE COMMISSION'S PROFILE REPLICATES EXACTLY — and the frame reproduces study 58

`perf record -e L1-dcache-load-misses:u -F 5999` on my own PGO build of `main`
at `450f766d` (`tools/perf60/pgo60.sh base`), same frame:

| function | commission | **this study** |
|---|---:|---:|
| `SotaHead::forward_into` | 20.44 | **21.55** |
| `avx512::i8_deltas_into_generic` | 17.44 | **17.55** |
| `HalfkaContext::current_full` | 17.08 | **16.96** |
| `ThreatContext::at_full` | 10.69 | **11.03** |
| `search::<NonPv>` | 10.81 | 9.51 |
| `HalfkaContext::finny_refresh_into` | 6.97 | 7.13 |
| `qsearch::<NonPv>` | 4.59 | 4.41 |
| `eval::evaluate` | 2.97 | 3.25 |
| `record_square_touch` | 2.51 | 2.64 |
| `init_quiets` | 2.44 | 2.36 |

The four named accumulator symbols are **63.19 %** against the commission's
61.93 %. Nothing in the brief's profile is wrong.

The frame also reproduces study 58 to five significant figures, which is what
makes every contrast below comparable to that document:

| | study 58 | this study |
|---|---:|---:|
| landed, non-PGO | 5819.69 i/n | **5819.68** |
| landed, PGO | 5729.02 i/n | **5728.98** |

Engine-level, non-PGO, from the §3 panel's 36 adjudicated rounds:
**`L1-dcache-loads` 2907.75/node, `L1-dcache-load-misses` 557.11/node — a
19.16 % miss rate**, against the brief's 543 and 18.33 % (PGO). Both readings
are right; they are different builds.

---

## ★ 2. THE BRIEF'S PREMISE IS WRONG IN ONE SPECIFIC WAY, AND IT DECIDES THE STUDY

> "This is the largest unworked block in the engine and nobody has attacked it
> as a *memory* problem."

**Studies 29, 39, 43, 48, 49, 50 and 51 are memory studies, and study 51
measured the ceiling and closed the routes to it.** The brief cites study 51's
prefetch catalogue but not its headline. Re-read before building anything:

**The ceiling (study 51 §2).** `probe-wpin` sends every folded row index to row
0, so the fold issues exactly the same load ops against a permanently
L1-resident row. It isolates *the entire cost of the weight rows being cold*,
including all collateral damage:

```
wpin51 vs base51, 15 rounds, byte-identical null in rotation
  cycles:u                    -13.186 %   se 0.212   15/15   p = 0.0001
  L1-dcache-load-misses:u     -20.456 %              15/15
  null: cycles +0.028 % (7/15, p = 1.00)
```

**13.19 % of cycles is the whole prize for the fold's weight rows, and it is a
LOWER bound** (wpin does 8.1 % more rows and 12.3 % more evaluations per node
and is still 13.19 % faster). Study 51 then decomposed it: **89.6 % is the fold
fetching its own bytes and 10.4 % is collateral pollution**, and Zen 4 offers no
pollution control at all on WB memory (`VMOVNTDQA` allocates in L1; `PREFETCHNTA`
does not de-prioritise — measured at the hardware level with a positive
control).

**And the routes to those bytes are individually closed, each with a number:**

| route | number that closed it | where |
|---|---|---|
| software prefetch of the rows | **eight** dead arms; ceiling is 1 of a row's 22 demand misses, and `T2` lands in L2 where they already hit 72.7 % | 4, 21, 29 ×3, 43 ×4, 44, 51 |
| skip all-zero rows | **30 of 19,616,896** folded rows (0.00015 %) | 51 §4.1 |
| nibble-pack narrow rows | **0.03 %** of weight-row bytes | 51 §4.2 |
| add/sub cancellation | **1.99 %** of threat rows, before paying ~10 comparisons per fold | 51 §4.4 |
| huge pages | 2.25 page walks/node total; arm built, verified, **null** | 37 §4 |
| break the 4 KB stack congruence | built (`cong51`), mechanism statistic **null**, panel on top of the null | 51 §4.3 |
| make the fold lazy | it already is; chain length **1.1346** against a floor of 1.0 | 51 §4.5 |
| accumulator-channel fusion | **+48 (loss) … −136 cycles/node**, and it trades away a refresh advantage SF does not have | 29 §4 — **DO NOT BUILD** |
| the pure-parent-copy threat fold | **already landed** as `alias[ply]` | 38 |

**The one framing error in the brief is the one study 29 §8 warns about in
capitals: "load volume is not load cost."** The brief ranks by share of
`L1-dcache-load-misses`. Study 51's per-region IBS census ranks by miss
*latency*, and the two orderings are not the same engine:

| region | miss rate | **share of engine miss latency** |
|---|---:|---:|
| `hka_weights` + `thr_weights` | 84.17 % / 73.64 % | **55.05 %** |
| `hka_stack` + `thr_stack` | 56.60 % / 45.21 % | 14.0 % |
| head tables (`l1_weights`, `l2_weights`, `skip_q`, `ft_scale`, `l3_weights`, `l1_biases`) | **91–100 %** | **11.8 %** |
| everything non-NNUE (`(other)` + `(stack/mmap)`) | 25.59 % / 4.55 % | 16.9 % |

`forward_into` is the brief's #1 by miss share and its tables are at 91–100 %
miss — and they are 11.8 % of miss latency, because they are small, dense,
sequential and their misses overlap. **A function closed for instruction work
can indeed be open for memory work, which is the brief's good point; but a
function at the top of a miss-COUNT ranking can also be near the bottom of the
cost ranking, and here it is.**

**The "≥47 % of cycles is stall" figure is an inference, not a measurement.** It
comes from assuming a Zen 4 peak IPC of 3–4 for a workload whose hot kernels are
512-bit AVX-512, where two 512-bit ops per cycle *is* the machine's ceiling. The
*measured* memory number on this engine is 13.19 % of cycles for the weight rows
(study 51, sign test 15/15) plus ~2–4 % for everything else in the table above,
against the brief's own ~9.4 % for branch mispredicts (not re-measured here).
Nothing in this study contradicts the brief's arithmetic; what it contradicts is
the reading that half the engine's time is recoverable by locality work.

**A second floor, computed here and not previously written down.** The fold's
own arithmetic is 2 vector ops per 32 lanes per row (`vpmovsxbw` folding the
32-byte load, then `vpaddw`/`vpsubw`), i.e. 80 512-bit ops per 1280-lane row. At
7.175 weight rows/node and Zen 4's 2 512-bit ops/cycle that is **~287
cycles/node ≈ 8 % of the frame that no memory work can touch**, because it is
issue, not stall. Weight-row misses (13.19 %) and this floor (8 %) together are
most of what the accumulator layer costs.

---

## ★ 3. ARM `regs4` — the REGS sweep's last untested setting. **LOSES.**

**The hypothesis, and it was a specific one.** The fold walks the accumulator in
tiles of `REGS` zmm. At the shipped `REGS = 5` a tile is 160 lanes, so a weight
row's per-tile slice is **160 bytes starting at byte offsets 0, 160, 320, …** —
and 160 mod 64 = 32, so four of the eight tile boundaries land *inside* a cache
line. Counting exactly: a row costs **24 line-touch events over 20 distinct
lines**. `REGS = 4` makes the slice 128 bytes at offsets 0, 128, 256, …, i.e.
**exactly two whole aligned lines per tile, 20 touches over 20 lines**.

Study 44 measured caps 20, 10, 8 and 5 and chose 5 on cycles — and **5 is the
only one of those that is NOT line-aligned** (640, 320, 256 are all multiples of
64). `regs4_44` has existed as a feature since study 44 and was never measured.
It is the last open setting, and the alignment argument is a mechanism study 44
did not have.

**Result — `docs/research/perf-artifacts-60/panel-regs4.txt`** (pre-rebase base
`450f766d`), 40 rounds (36 adjudicated), blocked Latin square, `null` a
byte-identical copy of `base` riding in the rotation, mean position 1.00 for all
three, node total 5,001,562 on every run, eval hash `2e1ede95…79c4` on every
binary. §5's post-rebase panel carries the same arm and agrees:

```
REFERENCE per node: instructions 5819.69  cycles 4962.93
                    L1-dcache-loads 2907.75  L1-dcache-load-misses 557.11

label   event                      median %     se    neg/n      p
null    instructions:u              -0.000   0.000    20/36   0.6177
null    L1-dcache-load-misses:u     +0.006   0.292    18/36   1.0000
null    L1-dcache-loads:u           +0.105   0.107    15/36   0.4050
null    cycles:u                    +0.012   1.581    18/36   1.0000

regs4   instructions:u              +1.537   0.000     0/36   0.0000
regs4   L1-dcache-load-misses:u     -0.024   0.285    18/36   1.0000
regs4   L1-dcache-loads:u           +0.380   0.093     7/36   0.0003
regs4   cycles:u                    +1.266   1.459    14/36   0.2430
```

**The arm's own mechanism reads dead.** `L1-dcache-load-misses` moves −0.024 %
at **18/36 — an exact coin flip** — against a null of +0.006 % at 18/36. The
four double-touched lines per row cost nothing, and the reason is now obvious in
hindsight and worth recording: their reuse distance is one tile of one fold,
about a kilobyte, so the second touch is an L1 **hit**. Removing a hit is not a
saving.

**And it costs.** `+1.537 % instructions` (0/36, zero spread) and `+0.380 %`
loads (7/36, p = 0.0003) against a load null of +0.105 %. Cycles is not
resolvable on this box today (§6) but its median has the same sign.

> **`REGS_CAP = 5` stands, and the sweep is now complete: 20, 10, 8, 5, 4 all
> measured. Do not re-open it.** Line alignment of the tile is worth nothing in
> this kernel; the tile width question is settled on the store queue, which is
> what study 44 said.

---

## ★ 4. ARM `hskip` — memory-level parallelism for the output-skip lane

**The mechanism.** `forward_into` ends with a strictly serial chain:
`l1_dot → l1_dequant → l2_tail → l3_dot → value`. The D47V output-skip lane —
`skip_dot_i8(ft, sk.q.block(bucket))`, a 1280-byte read that study 51 measured
at a **96.27 % L1 miss rate** — is computed at the *end* of that chain, where
there is nothing left to overlap its 20 cold lines with. But it depends only on
`ft` and on `bucket`, and both are ready the instant `activate_ft` returns.

`hskip60` issues it there instead. **Not a prefetch:** no hint instruction, no
extra load, the same loads issued ~800 instructions earlier. Bit-exact by
construction — the integer dot and the two f32 multiplies are unchanged and the
f32 *add* into `l3` stays exactly where it was, so no floating-point operation
is reordered. Node total and eval hash both confirm it.

**The inert-control trap this arm walked into, recorded because it is cheap to
repeat.** My first version hoisted `let bucket = OUTPUT_BUCKETS_LAYOUT[…]`
unconditionally so both arms could use it. That moved one index load across
`find_nnz` **in the arm-OFF build too**, and the control's `.text` md5 went
`d0dff381 → caab9574`. Every number would have been a property of the edit. The
binding is now duplicated under the two `cfg`s and the rebuilt control
reproduces `d0dff381` exactly — verified by building, not by mtime.

**Result — `docs/research/perf-artifacts-60/panel-post59.txt`**, base
`c09e21be`, 40 rounds, five labels in one blocked rotation, mean position 2.00
for all five, node total 5,001,562 on every run:

```
REFERENCE per node: instructions 5649.51  cycles 3565.16
                    L1-dcache-loads 2885.20  L1-dcache-load-misses 536.89

label   event                      median %     se    neg/n      p
null    instructions:u              -0.000   0.000    21/40   0.8746
null    L1-dcache-loads:u           -0.048   0.067    25/40   0.1539
null    L1-dcache-load-misses:u     +0.135   0.205    12/40   0.0166   ← fails
null    cycles:u                    +0.493   1.617    17/40   0.4296

hskip   instructions:u              -0.031   0.000    40/40   0.0000
hskip   L1-dcache-loads:u           -0.219   0.038    30/40   0.0022
hskip   L1-dcache-load-misses:u     +0.614   0.140     3/40   0.0000
hskip   cycles:u                    +0.711   1.291    12/40   0.0166
```

**The arm does what it was written to do and the mechanism goes the WRONG WAY.**

* `instructions:u` **−0.031 % = −1.75/node, 40 of 40 rounds at zero spread** —
  slightly *fewer* instructions, because the hoist puts `skip_dot_i8` adjacent
  to `find_nnz` and both stream the whole 1280-byte `ft`.
* `L1-dcache-loads` **−0.219 %** against a null of −0.048 %, i.e. **−4.9
  loads/node net**, 30/40, p = 0.0022.
* **`L1-dcache-load-misses` +0.614 % against a null of +0.135 %, i.e. +0.48 %
  net = +2.6 misses/node, 3/40, p < 0.0001.** The arm's whole premise was that
  those 20 cold lines are cheaper when issued early. They are not: **issued
  early they are resident *through* the sparse L1 matmul, competing with
  `l1_weights` for a 32 KB L1 that study 51 already measured at 91.44 % miss on
  that table.** Issued late they are fetched once, used once, and evicted with
  nothing left to displace.
* `cycles:u` is not resolvable (§6) — but its median has the same sign as the
  misses, not as the instructions.

> **Not landed.** Two counters improve, the one counter that states the arm's
> own mechanism gets worse, and the deciding counter cannot be read. Landing on
> the two that improved would be precisely the substitution the commission was
> written to stop. The feature stays in the tree with this number on it.

**The generalisable form, which is what this arm bought:** *hoisting a cold,
single-use read out of a serial tail and into a hot streaming phase does not
convert its misses — it adds the displaced data's misses to them.* Memory-level
parallelism is not free when the parallel work shares the cache.

---

## ★ 5. ARM `htail` — the ninth prefetch, and the first not aimed at the fold

**Why this is not prefetch kill #9 by construction.** All eight dead prefetch
arms target the fold's weight rows, where study 29 §5.2 computed the ceiling
from first principles: a row is a cold single-visit burst taking 22 demand
misses, one hint converts at most 1 of 22, and SF's `PrefetchLoc::LOW` = `T2`
lands the line in L2 — where those rows already hit 72.7 % of the time.

**None of those three facts holds for the head's tail tables.** They are
**2,368 bytes** at this arch (`L2 = 16`, `L3 = 32`: `l2_weights` 16·32·4 = 2048,
`l2_biases` 128, `l3_weights` 128, `l1_biases` 64 — which is also the
arithmetic that reproduces study 51's "`l1_weights` 20 KB/bucket" and its
"`l2_weights` 2048 B"), they
are read **in full on every evaluation** so a hint cannot be spent on data that
is not touched, study 51's region census puts them at **98.84 % / 100.00 % L1
miss**, and their addresses are known at function entry while their first use is
~800 instructions away at the end of the serial chain. `T0`, not `T2`, because
the miss being converted is an L1 miss.

Cost is the thing to watch: 36 hints per evaluation.

**Result**, same panel, same 40 rounds, same null:

```
label   event                      median %     se    neg/n      p
htail   instructions:u              +0.404   0.000     0/40   0.0000
htail   L1-dcache-loads:u           +0.205   0.046     6/40   0.0000
htail   L1-dcache-load-misses:u     +0.379   0.096     5/40   0.0000
htail   cycles:u                    +0.046   1.098    19/40   0.8746
```

**+0.404 % instructions = +22.8/node** (predicted 15.5 from 36 hints ×
0.43114 evaluations/node; the extra is the address arithmetic LLVM emits for
the block pointers). And **the misses it exists to convert went UP**: +0.379 %
against a null of +0.135 %, i.e. **+0.24 % net, 5/40, p < 0.0001**.

**Prefetch kill #9**, and the informative part is that it died at the one site
where study 29 §5.2's ceiling argument explicitly did *not* apply. The
mechanism it refutes is the general one: the tail tables' 36 lines are read as
one dense sequential burst whose misses already overlap each other, so there
was no exposed latency for a hint to hide — only 36 extra address computations
and 36 extra L1 lookups. The rule that now covers all nine: **a hint pays only
where the miss is both cold and SERIALISING; a burst of independent cold misses
is already as parallel as the machine can make it.**

---

## ★ 5.1 `tools/wallclock_panel.py` — RUN, AND IT REFUSED

The commission requires the wall-clock panel and says an arm is not valuable
without one. It was run, on the largest contrast this study has — **the §0 PGO
pair, whose instruction difference is +5.44 %**, far bigger than any arm — so
that a refusal could not be blamed on the effect being small:

```
python3 tools/wallclock_panel.py --base bin60/base-pgo --arm bin60/nonetpgo --rounds 24

NULL (byte-identical): median -0.376%  sd 1.137  sign 17/24 neg  p=0.0639
ARM                  : median +0.354%  sd 0.941  sign  8/24 neg  p=0.1516

NOT LICENSED: null median -0.376% vs arm +0.354%.
```

**The pre-registered licence fired and the arm is not quotable.** Two things go
in the record anyway, both about the instrument rather than the arm:

1. **A byte-identical null moved −0.376 % with sd 1.137 over 24 rounds.** That
   is the wall-clock analogue of §6's `cycles:u` spread and it is why no arm in
   this document carries a value claim. The refusal is the tool working.
2. **The arm's median was +0.354 % for a +5.44 % instruction difference.** It
   is NOT a licensed number and must not be quoted as one — but it is the same
   direction, and roughly the same magnitude of disconnect, that the commission
   itself measured across studies 52–58 (−486.8 i/n for ~−0.9 % wall-clock).
   Whoever gets a quiet box should re-run exactly this pair first: it is a large,
   cheap, node-identical contrast that calibrates the whole instructions →
   time conversion, and it costs six minutes.

---

## ★ 6. THE INSTRUMENT: the same null moved 4.6× between panels, on the same box

Every counter quoted above is quoted against a null that rode in the same
rotation, and the two panels' nulls are not the same instrument:

| counter | pre-rebase panel (§3, n = 36) | post-rebase panel (§4–§5, n = 40) |
|---|---|---|
| `instructions:u` | −0.000 %, se **0.000**, 20/36 | −0.000 %, se **0.000**, 21/40 |
| `L1-dcache-loads:u` | +0.105 %, se 0.107, 15/36 | −0.048 %, se 0.067, 25/40 |
| `L1-dcache-load-misses:u` | +0.006 %, se 0.292, **18/36 (passes)** | +0.135 %, se 0.205, **12/40, p = 0.0166 (FAILS)** |
| **`cycles:u`** | +0.012 %, se **1.581**, 18/36 | +0.493 %, se **1.617**, 17/40 |

**`cycles:u`'s per-round spread is 1.6 % in both — five times the standing
quiet-box σ of 0.25–0.30 %**, so no arm here can be adjudicated on it. The
cause is external load throughout: perf study 59's builds, a 1500-game match,
a GPU training run, and — for the second panel's later rounds — **this study's
own `check.sh` gate, which I deliberately ran in parallel to save wall-clock
time and which cost me the one quiet window the box offered.** An interim read
of the same panel at n = 15, taken before the gate started, had `cycles:u`'s
null at **se 0.343, 7/15** — usable. That window closed because I filled it.

**`L1-dcache-load-misses` fails its null in the second panel (12/40,
p = 0.0166) and passes in the first (18/36).** Every miss figure in §4 and §5 is
therefore quoted **arm minus null**, which is what makes `hskip`'s +0.48 % and
`htail`'s +0.24 % net rather than raw.

`main`'s own handoff `c09e21be`, written the same afternoon, reached the
identical conclusion from the other side: *"cycles:u is NOT load-robust, and
the sigma = 1.57 % figure measured on a quiet box understates it by more than an
order of magnitude under real contention. instructions:u is load-robust, which
is exactly why it is the gate."* It voided study 59's own cycles claim on that
basis. **Two independent measurements of the same defect on the same day.**

**Consequences, applied rather than noted:**

* §3's `regs4` verdict rests on `instructions:u` (+1.537 %, zero spread) and on
  a **dead mechanism statistic** — not on its cycles column, which is quoted
  only for its sign.
* §4 and §5 are adjudicated on `instructions:u` (exact) and on the
  arm-minus-null miss contrast, never on cycles.
* `tools/wallclock_panel.py`'s pre-registered licence (null median under half
  the arm) is the same test one level up, and §5.1 reports what it returned.
* **The standing σ = 1.57 % figure should be re-labelled a QUIET-BOX floor.**
  It is not a property of the box; it is a property of the box *when nothing
  else is running*, and this engine's own perf programme is usually running
  something else.

---

## ★ 7. WHAT THE NEXT STUDY SHOULD KNOW

1. **The weight rows are 55 % of engine miss latency and 13.19 % of cycles, and
   nine routes to them are closed with numbers.** The table in §2 is the
   catalogue. Do not re-derive it; extend it.
2. **Rank memory work by miss LATENCY, not miss COUNT.** `forward_into` is #1 on
   the count ranking and 11.8 % of the latency; the two accumulator stacks are
   14 % of latency and study 29's `accpin` showed their READ half is nearly
   free. The census that gives this exists: `--features foldcount` plus
   `tools/perf51/ibsaddr51.py`.
3. **`REGS_CAP` is closed at every setting.** 20/10/8/5 (study 44) and 4 (§3).
   Line-aligning the tile's row slice — the one mechanism study 44 did not
   have — is worth **nothing**, because the lines a misaligned tile touches
   twice are touched a kilobyte apart and the second touch is an L1 hit.
4. **The fold has an arithmetic floor of ~8 % of the frame** (§2, 2 vector ops
   per 32 lanes per row × 7.175 rows/node ÷ 2 512-bit ops/cycle). Any proposal
   that claims to make the accumulator layer much cheaper without changing the
   architecture is claiming to beat that, and should be asked to say how.
5. **Two new rules about hints and hoists, each paid for by an arm here:**
   *a hint pays only where the miss is cold AND serialising* (§5 — a dense
   burst of independent cold misses is already maximally parallel, and 36 hints
   bought +22.8 i/n and MORE misses); and *hoisting a cold single-use read into
   a hot streaming phase adds the displaced data's misses to its own* (§4 —
   `hskip` reduced instructions and loads and raised misses 0.48 %).
5. **Never run a perf panel while another study builds.** `cycles:u` se went
   from the standing 0.25–0.30 % to 1.58 %. The build lock protects memory, not
   measurement; there is no lock for the box being quiet, and this study needed
   one.
6. **Check the control's `.text` md5 after every source edit, including edits
   that are entirely inside a `#[cfg]`.** §4 changed the arm-OFF build by moving
   one `let` outside the `cfg`.
7. **Do not run `git stash`, `git checkout --` or a rebase in a tree while a
   panel is appending to a file inside it.** I stashed the working tree to
   rebase, git reverted `panel-head.txt` to its committed 10-record version
   *underneath the running panel*, and the panel carried on appending. That is
   study 57 §2's two-writers-one-file hazard arriving from a completely
   different direction — the `flock` guard cannot see a writer that is not
   another panel. **The panel was killed by PID and the run discarded**, not
   repaired: a file whose middle was replaced is not a measurement. Rerun cost
   ~20 minutes.
7. **Still open, best-priced first, all inherited:** study 43 §9 item 2's
   subtractive Finny-line arm; study 57 §6's FNV-1a-64 payload checksum
   (~542 M cycles ≈ 45 % of all startup, needs a format ruling); study 48 §9
   item 3's AVX2 `activate` hoist. And the one genuinely unidentified block in
   the engine: **study 51 §3's "what evicts the head's tables" — `search::<NonPv>`
   is 23 % of cycles and its stall mechanism has never been named**, with study
   31 having refuted the one candidate anybody proposed.

---

## ★ 8. REPRODUCE

```bash
ln -s /path/to/chessengine/data data          # the worktree needs the net
bash tools/perf60/build60.sh base             # non-PGO native control
cp bin60/base bin60/null                      # the byte-identical null
bash tools/perf60/build60.sh regs4 regs4_44
bash tools/perf60/build60.sh hskip hskip60
bash tools/perf60/build60.sh htail htail60
bash tools/perf60/pgo60.sh   base             # native+PGO, needs the net
bash tools/perf60/ident60.sh base regs4 hskip htail    # nodes + eval hash

P60_PANEL=docs/research/perf-artifacts-60/panel-regs4.txt \
  bash tools/perf60/panel60.sh 40 base null regs4
python3 tools/perf60/med60.py docs/research/perf-artifacts-60/panel-regs4.txt \
  base null regs4

P60_PANEL=docs/research/perf-artifacts-60/panel-head.txt \
  bash tools/perf60/panel60.sh 40 base null hskip htail
python3 tools/perf60/med60.py docs/research/perf-artifacts-60/panel-head.txt \
  base null hskip htail

bash tools/perf60/pair60.sh 3 base base-pgo nonetpgo   # the §0 PGO contrast
CHECK_NNUE_GATE=1 bash tools/check.sh
cargo test --release -- --ignored perft_full_suite
```
