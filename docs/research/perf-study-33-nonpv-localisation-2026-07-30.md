# ARCHON perf study #33 — localising `search::<NonPv>`: the site is average, and two thirds of it is not search

Date: 2026-07-30. Instruments: `tools/perf33/`. Artifacts:
`docs/research/perf-artifacts-33/`. Commits on this branch: `f753d74`,
`0e0d217`, `4a0446b`, `aff8939`, `a7d3d9f`, and this write-up.

**Commission:** `search::<NonPv>` is 19.90% of ARCHON's profile and study 28
measured **+303.4 cycles/node** there (26.7% of the whole Stockfish gap). Study
31 built an arm against its only named mechanism and got a **measured null**.
Find the mechanism.

**What this study did not do: land a speedup.** §9 says what I tried, why I
killed it before writing it, and what the data says the next arm should be.
That is a real failure against the commission and it is stated here rather than
buried.

---

## 0. HEADLINE

### 0.1 The five results

1. **The site reproduces, and the statistic that measures it has a 0.80%
   null.** 410.5 stall cycles/node in the probe frame. Not a build artifact.
2. **"19.90% of the profile" is roughly half an inlining boundary.** Same
   source, PGO on vs off: 410.5 → 759.7 cycles/node while `Worker::do_move`
   collapses 116.6 → 18.0. PGO folds make/unmake and the accumulator push into
   the symbol. Decomposed by DWARF inline frame, **only 32.87% of the PGO
   symbol is search's own code** (§4).
3. **Only 14.5% of it is NNUE shadow, measured causally.** Study 29's
   `probe-wpin` removes every weight-row miss in the engine — 40.5% of all
   stall cycles — and takes 14.5% of this symbol's with it (§3.3).
4. **The site is not a locality defect. It is exactly average.** Its share of
   the engine's load-stall cycles (8.80%) equals its share of the engine's
   retired load ops (8.84%) to two decimal places, while it carries 1.69% of
   the engine's misses and 91.4% of its loads hit L1 (§5). **The cost tracks
   the NUMBER of loads the node issues.** No locality fix can reach it.
5. **`search::<NonPv>` is the engine's #1 symbol for branch mispredicts** —
   17.47%, 4.3 per node against an engine 24.1 — which nobody had looked at.
   Also distributed: the largest single site is 0.39 mispredicts/node (§6).

### 0.2 What replaces study 28's item

Study 28's chain was: the symbol stalls 3.03× Stockfish's ⇒ something at this
site is cold ⇒ it is the per-node snapshot into a 10.5 KB-strided stack. Study
31 killed the third link. This study kills the second and rebuilds the first:

> The symbol is large because **the node issues a lot of loads**, and it stalls
> **exactly the engine-average number of cycles per load issued**. Two thirds
> of the symbol is inlined callees that other builds report separately, so
> "3.03× Stockfish's search symbol" is substantially a comparison of two
> different amounts of inlined code — which is the caveat study 28 wrote for
> Reckless and for Stockfish, and did not apply to ARCHON's own PGO build.

**This is not "it is diffuse".** Diffuse would be an absence of structure. The
structure is measured and it is a proportionality: stall/load index **1.00**,
against 0.28 for the overlapped streaming kernel `forward_into` and 6.60 for
the genuinely serial `apply_at`. The node is neither pathological nor
efficient; it is average, and the only lever on an average site is its volume.

---

## 1. METHOD

### 1.1 Frame

Identical to studies 29 and 31 so the numbers compose:
`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
— 50 FENs × 100 k nodes, **5,001,562 nodes**, one thread, 16 MB TT, D48X.
Release profile (fat LTO, CGU = 1), `target-cpu=native`, `--features
sota,arch-d48x`. Non-PGO unless the row says PGO.

**Per-node costs come from the sample period, not a second run.** `perf record
-c <period>` makes every sample stand for exactly `period` events, so per-node
cost at a symbol = samples × period / nodes. All 706,794 IBS samples carried
period 50021 — checked, not assumed. This removes the class of error where a
separate `perf stat` run drifts from the `perf record` it is normalising.

### 1.2 THE BOX WAS NOT QUIET, and here is what that costs

The D49 trainer ran throughout at ~143% CPU, and perf study 32 was recording on
CPU 6 in a sibling worktree for the whole session. I pinned every measurement
to **CPU 2** (SMT sibling 10) and every build to cores 0,1 at nice 19, but the
L3 is shared and that is not fixable by pinning.

The visible consequence: my engine total is **4665.4 stall cycles/node against
study 28's 2458.6** on a quiet PGO box — 1.9×. So:

* **Absolute per-node numbers in this document are inflated and are only
  comparable WITHIN it.** The apparent agreement between my 410.5 and study
  28's 452.6 is partly luck: the two frames differ by 1.9× in the denominator.
* **Shares and paired ratios are the defensible quantities**, and every claim
  in §0 is one of those. The like-for-like comparison to study 28 is
  share-to-share on a PGO build: **16.04% mine vs 18.41% theirs** (§3.2).

### 1.3 Null first — and the instrument is provably inert

The whole study reads per-source-line and per-inline-frame attribution off a
build carrying DWARF line tables (`--profile profiling`, which inherits release
and adds `debug = "line-tables-only"`). Everything depends on line tables not
changing codegen. That is not asserted from documentation; it is measured, by
`tools/perf33/textnull33.py`:

| check on `search::<NonPv>`, release vs profiling | result |
|---|---|
| instruction count | 3326 = 3326 |
| per-instruction offsets | **0 differing** |
| instruction lengths | **0 differing** |
| mnemonics, in order | **0 differing** |
| instructions differing in bytes | 39, **all** inside one RIP-relative displacement field |
| those displacement deltas | ±32 exactly = the whole-symbol `.text` shift |

**Raw `.text` bytes do NOT match and that is a red herring** that cost me time:
the two cargo profiles get different `-C metadata`, every v0 symbol carries a
different crate disambiguator, and the symbol-table size difference shifts
`.text` by 32 bytes. A naive `cmp` reports 9134 differing bytes for zero
codegen difference. Anyone repeating this should compare instruction streams,
not section bytes.

**Run-to-run null on the load-bearing statistic.** A byte-identical copy of the
probe binary, recorded as its own arm: `search::<NonPv>` 410.5 → 413.8,
**+0.80%**; engine total 4665.4 → 4735.7, +1.51%. So the −14.5% in §3.3 is 18×
its null and the +85.1% in §3.2 is 106× it.

### 1.4 The second null, which caught a real bug — see §8

`perf script -F sym` returns `[unknown]` for IBS records on this perf build, so
the IBS tools resolve IPs against `nm` ranges themselves. A home-made resolver
is exactly the instrument that fails silently, so it is cross-checked against
`perf report` on the same file (`tools/perf33/ibsnull33.py`). It failed at
first — 13.50 percentage points — and after the fix agrees to **0.00 points**
over 14 symbols. §8 is the retraction.

---

## 2. WHAT THE SYMBOL IS

`search::<NonPv>` is **15,898 bytes / 3,326 instructions** in the probe build.
That is why a symbol share cannot explain itself: LTO inlines the TT probe,
make/unmake, the accumulator push, the move picker's `next_move`, legality,
SEE and the history updates into one name.

`tools/perf33/inline33.py` splits it by reconstructing each sample's full DWARF
inline stack and aggregating two ways — by the **innermost** frame (whose code
the sample is in) and by the **outermost frame below `search`** (which phase of
the node it belongs to).

---

## 3. THE THREE ARMS

All on the load-stall event `ex_no_retire.load_not_complete:u`, period 50021.

| arm | `search::<NonPv>` | vs base | engine total | vs base |
|---|---:|---:|---:|---:|
| base (probe build) | **410.5** | — | 4665.4 | — |
| **NULL** — byte-identical copy | 413.8 | **+0.80%** | 4735.7 | +1.51% |
| `probe-wpin` (study 29's) | **351.1** | **−14.47%** | 2774.6 | **−40.53%** |
| PGO, same source | **759.7** | **+85.06%** | 4737.5 | +1.55% |

### 3.1 The site reproduces

410.5 stall cycles/node against study 28's 452.6, on a statistic whose
run-to-run null is +0.80%. Whatever else is true, study 28 did not measure an
artifact of its own build.

### 3.2 …but its SHARE is half an inlining boundary

Same source, PGO on: the symbol goes 410.5 → 759.7 cycles/node, **+85%**, while
`Worker::do_move` — a separate symbol in the probe build at 116.6 cycles/node —
collapses to 18.0. PGO inlines it.

As shares of the engine's stall cycles:

| build | `search::<NonPv>` share |
|---|---:|
| probe (non-PGO), this study | **8.80%** |
| PGO, this study | **16.04%** |
| PGO, study 28 | 18.41% |

The PGO figures are the like-for-like pair and they are close; the 2.4-point
residual is a different PGO training workload and a contended box. **The
8.80% ↔ 16.04% gap is not noise and not the box — it is the compiler putting
more code inside the symbol.** Study 28 wrote exactly this caveat about
Reckless ("70.2% of its instructions land in `search::<NonPV>` because it
inlines make-move, the move picker and the whole `BoardObserver` threat-event
path") and about Stockfish splitting out `update_all_stats`; it did not apply
it to ARCHON's own PGO binary, and the 3.03× ratio inherits the omission.

### 3.3 Only 14.5% of the site is NNUE shadow — a causal test

`probe-wpin` makes every accumulator fold read weight row 0 whatever the
feature index says: identical load count, identical widths, identical dispatch
shapes, and the entire 57.7 MB HalfKA and 77.7 MB threat tables collapse to one
hot row. It removes **40.53% of the engine's stall cycles** and **14.47% of
this symbol's**, against a +0.80% null.

So NNUE weight streaming does evict the search node's working set, and that is
worth about **59 cycles/node** here. It is a real mechanism and it is the first
time this interference has been priced at this site. But **85.5% of the site
survives with every weight-row miss in the engine deleted.** The site is not
downstream of the NNUE's memory traffic.

(Node counts: base 5,001,562, wpin 5,001,623. `bench … nodes` fixes the budget;
wpin computes wrong evaluations on purpose, so the trees differ marginally.)

---

## 4. WHAT THE 19.90% IS MADE OF

PGO build, 75,963 samples in the symbol, 759.7 stall cycles/node, aggregated by
outermost inline frame below `search`:

| phase | share | cycles/node |
|---|---:|---:|
| **search's OWN code** | **32.87%** | **249.7** |
| **`do_move`** (make_move + accumulator push) | **26.12%** | **198.4** |
| `next_move` (move picker) | 6.38% | 48.5 |
| `legal` | 4.96% | 37.7 |
| `all_threats` | 4.55% | 34.6 |
| `undo_move` | 3.56% | 27.0 |
| `probe` (TT) | 3.17% | 24.1 |
| `correction_value` | 2.91% | 22.1 |
| `see_ge` | 2.79% | 21.2 |
| `gives_check` | 1.13% | 8.6 |
| history updates (`update_all_stats` + gravity) | 1.92% | 14.6 |
| `evaluate` | 0.66% | 5.0 |

**Two thirds of the largest symbol in ARCHON's profile is not search.** That is
the commission's question — "what fraction is movepick, TT probe, eval call,
make/unmake, pruning arithmetic?" — answered.

### 4.1 The top source line, and why it is not what it looks like

The symbol's hottest single line is **`sota_accumulator.rs:343` at 11.25%
(85.5 cycles/node)**, which is

```rust
self.meta[ply + 1] = Meta { dirty, computed: [false; 2] };
```

— the accumulator push, i.e. study 28's item C **by name**. Study 31 falsified
the locality version of that item on the **threat** side (`ThreatEntry`, a
5,312-byte stride). This is the HalfKA side, and it is a different array. So it
is worth saying plainly what it is not:

`meta` is already a dense `Box<[Meta; STACK_LEN]>`. **There is no stride to
fix.** And line 343 is a **store**, which cannot itself wait on load data —
`ex_no_retire.load_not_complete` charges the cycle to the oldest un-retired
instruction, so this line is accumulating the wait for loads issued *earlier*,
in `DirtyPiece::from_move` and `make_move`. The #2 line, `core/src/bool.rs:66`
at 10.23%, is the same story from the other side: it is the closure body of

```rust
let dirty = self.eval_ctx.nnue_active()
    .then(|| DirtyPiece::from_move(&self.board, m));
```

**Standing methodological note: a non-precise cycle-counting event names where
the ROB head parks, not where the cost is.** Two of the three hottest "lines"
in the biggest symbol in the engine are parking spots. This is why §5 uses IBS
instead.

---

## 5. THE MECHANISM: THE SITE IS AVERAGE PER LOAD

IBS tags a **retired** op with its own IP, its data address and the level that
served it. 706,794 samples, 72,992 in the symbol (10.33%, matching `perf
report` exactly), 20,707 of them loads.

### 5.1 The loads inside the symbol are hot

| level | share of the symbol's loads |
|---|---:|
| L1 hit | **91.4%** |
| L2 hit | 8.3% |
| L3 hit | 0.3% |
| DRAM | 0.0% |

The symbol holds **1.69% of the engine's non-L1 loads** against **8.80% of its
stall cycles**. Whatever the site costs, it is not cache misses.

### 5.2 The statistic that names it

Two indices, both ratios of shares — deliberately, because `L1-dcache-loads`
double-counts a 512-bit load on Zen 4 (2.0015 events per `vmovdqu64`, study 29)
and any comparison of a scalar symbol to an AVX-512 kernel through a perf-stat
load total would be wrong by ~2× in the direction that flatters this study.
IBS counts retired ops, so both shares sit on the same footing.

| symbol | %stall | %load ops | %misses | **stall/load** | stall/miss |
|---|---:|---:|---:|---:|---:|
| `ThreatContext::at_full` | 22.07% | 22.49% | 28.01% | 0.98 | 0.79 |
| `avx512::i16_deltas_into` | 15.82% | 15.03% | 27.29% | 1.05 | 0.58 |
| `finny_refresh_into` | 13.03% | 14.31% | 26.90% | 0.91 | 0.48 |
| **`search::<NonPv>`** | **8.80%** | **8.84%** | **1.69%** | **1.00** | **5.20** |
| `MovePicker::init_quiets` | 5.77% | 4.84% | 0.84% | 1.19 | 6.84 |
| `HalfkaContext::current_full` | 4.39% | 2.13% | 0.42% | 2.06 | 10.47 |
| `ThreatContext::apply_at` | 3.08% | 0.47% | 0.03% | **6.60** | 104.86 |
| `SotaHead::forward_into` | 2.64% | 9.61% | 8.66% | **0.28** | 0.31 |

`search::<NonPv>` stalls at index **1.00** — its share of the engine's stall
cycles equals its share of the engine's retired load ops to two decimals.

The two extremes calibrate it. `forward_into` at **0.28** carries 8.66% of the
engine's misses and almost no stall: a streaming kernel with many misses in
flight, already overlapped — which is the same conclusion study 29 reached from
a different instrument ("12.25% of the engine's loads but only 2.18% of its
load-stall cycles"), reached here independently. `apply_at` at **6.60** is what
a genuinely serial site looks like.

**The search node is neither.** It is the engine's average, and it is large
because it issues a lot of loads.

### 5.3 What that licenses, and what it forbids

* **Forbidden:** any locality intervention at this site. Its loads already hit
  L1 91.4% of the time and it takes 1.69% of the engine's misses. Reordering,
  re-striding or densifying the structures the node touches has almost nothing
  to work on. This is the same verdict study 31 got from an actual arm, now
  with a reason attached, and it should stop the item being proposed a third
  time.
* **Licensed:** reducing the **number of loads** the node executes. At index
  1.00 the payoff is linear and predictable: −10% of the node's load ops is
  −10% of its stall cycles, ~41 cycles/node in the probe frame.
* **Also licensed:** the 14.5% NNUE-eviction component (§3.3), which is not
  addressable from inside search but is a real cost of the weight-row traffic
  and belongs on the NNUE side of the roadmap.

---

## 6. BRANCH MISPREDICTS — the axis nobody had looked at

`ex_ret_brn_misp:u`, period 2003.

| | mispredicts/node | share |
|---|---:|---:|
| **`search::<NonPv>`** | **4.25** | **17.47%** |
| `MovePicker::init_quiets` | 3.3 | 13.6% |
| `ThreatContext::at_full` | 2.6 | 10.7% |
| `Worker::do_move` | 1.4 | 5.8% |
| engine total | 24.1 | — |

**It is the engine's #1 symbol on this event** — a completely different ranking
from the stall profile, where it is fourth. At a ~15–20 cycle penalty, 4.25
mispredicts/node is roughly 65–85 cycles/node, a separate bucket from the
load-stall cycles above.

Localised, it is distributed like everything else here. The largest single site
is `movepick.rs:531` — the staged picker's `match self.stage` — at 9.12% of the
symbol's mispredicts, i.e. **0.39 mispredicts/node ≈ 6–8 cycles/node**. There is
no single branch worth converting to arithmetic.

---

## 7. THE INSTRUMENTS

`tools/perf33/`, all reusable, each with the null it needs:

| tool | what it does |
|---|---|
| `build33.sh` / `buildpgo33.sh` | the two build paths (probe, and PGO **with line tables**) under this box's memory discipline |
| `textnull33.py` | **the null** — proves the line-table build is codegen-identical, instruction by instruction |
| `rec33.sh` / `ibs33.sh` | record one event / IBS ops with data source, pinned off study 32's core |
| `inline33.py` | split a symbol into its DWARF inline frames (innermost, and phase) |
| `pernode33.py` | sampled profile → per-node event counts, several arms side by side |
| `ibsline33.py` / `ibsmiss33.py` | per-load data-source census inside a symbol, by line and by phase |
| `ibsnull33.py` | **the null** — cross-checks the home-made IP→symbol resolver against `perf report` |
| `elfmap33.py` | runtime IP → ELF vaddr, reading `p_vaddr − p_offset` from the program headers |
| `mlp33.py` | the stall/load and stall/miss indices of §5.2 |

---

## 8. RETRACTION

**Commit `0e0d217` published engine-wide IBS load numbers computed with a
symbol map that was one page off.** Retracted by `a7d3d9f`.

`perf script` cannot symbolise IBS records on this perf build, so the tools
resolve IPs themselves. The MMAP2 record gives a mapping's runtime base and its
**file offset**; `nm` reports **ELF virtual** addresses. For this binary the
executable LOAD segment sits at file offset `0x3e150` and virtual address
`0x3f150` — they differ by `0x1000`, because the linker page-aligns a segment's
vaddr independently of its offset. I used `ip − vbase + pgoff`.

It resolved every sample to *a* symbol, produced a plausible table, and summed
to 100%. What caught it was running the null: my resolver said
`i16_deltas_into` held 0.30% of IBS samples where `perf report` said 13.80%.
After the fix the worst disagreement over 14 symbols is **0.00 points**.

| number | as published | corrected |
|---|---:|---:|
| loads in symbol, L1 hit | 90.5% | **91.4%** |
| symbol's share of engine non-L1 loads | 1.08% | **1.69%** |
| symbol's share of IBS samples | 6.92% | **10.33%** |

The conclusion of that commit survives — the site's loads are hot, and it is
not where the engine's misses are — and the corrected table made §5.2's index
possible, which the wrong one did not support. **The per-symbol table in that
commit was badly wrong and must not be quoted.**

Unaffected: everything from `inline33.py` and `pernode33.py`, which use perf's
own `-F ip,sym,symoff` resolution. That is §3, §4 and §6 — the study's
headline results.

---

## 9. WHAT I TRIED AND KILLED, AND WHAT THE NEXT ARM IS

**No speedup landed.** One arm was designed and killed before implementation,
and I record it because the reason is the useful part.

**Killed arm — skip `correction_value` on in-check nodes.** It is computed
unconditionally at `search/mod.rs:2356`, reads 6–8 random entries out of the
1 MB `corr` and 1.22 MB `cont_corr` tables, and its first two consumers
(`to_corrected_static_eval` at 2514, razoring at 2776) are both inside
`if !in_check`. Value-identical, three lines, exactly gated by `Bench: 587683`.
**It is wrong:** an exhaustive grep — not the truncated one I ran first — shows
`correction_value` is also read at **3459** (`corr_adj`, the singular-extension
margin) and **3590** (the LMR reduction), both inside the move loop, which
in-check nodes reach. The value is live on every node. Killed at zero cost
because I re-ran the grep without `| head`.

**Considered and not attempted, with reasons:**

* *Move `cont_hist` out of `Stack` into a dense `[ContHistId; 256]`.* The six
  reads `stack[cur-1..=cur-6].cont_hist` are 36–40 bytes apart, spanning 4–5
  lines; dense they would be 12 contiguous bytes. But it is a 42-site refactor
  across `&[Stack]`/`&mut [Stack]` signatures, and §5 says the site has almost
  no misses to recover. **This is the same class of change study 31 measured as
  a null**, and nothing in my data promotes it.
* *Drop the dead `MovePicker.cont_hist[4]`.* Real — offset 5 is never read —
  but `movepick.rs:762` already documents it as deliberate SF parity, and the
  win is one stack load per node. Renumbering the array for that is bug risk
  without payoff.

**What the next arm should be, from §5.3.** The site's cost is linear in its
load-op count, so the item is *fewer loads per node*, not better-placed ones,
and the biggest single addressable block is §4's `do_move` at **198.4
cycles/node / 26.12% of the symbol** — whose two hottest lines are both
`DirtyPiece::from_move` deriving mover, victim, ep square and castling rook
geometry from the pre-move board immediately before `make_move_with` re-derives
the same facts. `search/mod.rs:4851` records that perf study 13 measured and
rejected fusing them at **+41.5 i/n** — an *instruction-count* rejection, which
is exactly the currency study 20 proved wrong (−5.99% cycles with instructions
up 0.19%). **That item deserves re-measurement on cycles and on load ops, and
it is the one I would run next.**

---

## 10. WHAT I AM LEAST SURE ABOUT

In order.

1. **That the stall/load index of 1.00 means what I say it means.** It is one
   number from two profiles taken in separate runs on a contended box, and I
   did not run a null on the index itself (i.e. two independent base runs
   through `mlp33.py`). The individual inputs each have nulls — the stall
   profile at +0.80%, the IBS resolver at 0.00 points — but their *ratio* does
   not. It landing on 1.00 to two decimals is suspiciously tidy for a
   contended box, and §0 rule 6 says to attack that rather than enjoy it. The
   calibration extremes (0.28 and 6.60) are what make me believe the index has
   dynamic range at all; without them I would not quote it.
2. **The PGO comparison rests on ONE PGO build with my own training
   workload**, not study 28's. The 16.04% vs 18.41% agreement is one point
   against one inherited point, and PGO inlining is precisely the thing that
   varies with the training workload. The *direction* (PGO folds `do_move` in,
   nearly doubling the symbol) is solid because `do_move`'s collapse
   116.6 → 18.0 is visible in the same table; the *magnitude* of "half the
   19.90%" is one measurement.
3. **Everything absolute is inflated ~1.9× by the box** (§1.2). I believe the
   ratios and the shares. I would not defend 410.5 or 759.7 as numbers a quiet
   box would reproduce.
4. **`probe-wpin` searches a marginally different tree** (5,001,623 vs
   5,001,562 nodes) because it computes wrong evaluations by design. At −14.47%
   against a +0.80% null the sign is not in doubt, but the magnitude carries
   that confound and study 29 flagged it too.
5. **Whether the branch-mispredict bucket is worth more than I implied.** I
   priced 4.25/node at a textbook 15–20 cycle penalty rather than measuring the
   penalty on this machine, and I did not check how much of it overlaps with
   the load-stall cycles rather than adding to them. It is the least-examined
   number in the document and it is the one I would instrument next after the
   `do_move` item.
