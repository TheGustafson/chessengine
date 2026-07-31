# Perf study 52 — `search::<NonPv>` decomposed: 19.90% is not a number, and the symbol is the engine's most op-efficient large site

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at `af7b8e19` (studies 49, 50,
51 merged) · **Net/arch:** D48X @ `network_scale=600`, `sota,arch-d48x`,
ACC 1280 · **Instruments:** `tools/perf52/` · **Artifacts:**
`docs/research/perf-artifacts-52/` · **Branch:** `perf-study-52-nonpv`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, pinned to
physical core 2 (siblings 2,10). Two binaries, both node-identical:
`base52` (shipped release profile, `-C target-cpu=native`, non-PGO) and
`pgo52` (PGO **plus** DWARF line tables, study 33's `buildpgo33.sh` recipe —
the only build in which the commissioned 19.90% can exist at all).

---

## ★ 0. THE ANSWER, FIRST

### 0.1 The commissioned number does not reproduce, because it is not a property of the engine

**`search::<NonPv>`'s share of cycles is 12.33% or 25.38% depending on nothing
but which compiler pipeline built the binary.** Both were measured tonight, in
the same frame, from the same source, on an idle box, with startup symbols
(`load_v2_bytes`, `quantise_halfka`, `init_magics`) excluded from the
denominator:

| build | `search::<NonPv>`, % of ENGINE cycles |
|---|---:|
| `base52` — shipped release profile, non-PGO | **12.33 %** |
| `pgo52` — PGO, same source | **25.38 %** |

The commission's **19.90 % is neither, and it is stale**: it originates in study
28, was carried by study 33, and **study 36 §0.2 already superseded it at
9.80 %** on a different (heavily loaded) box. No study since 36 published a
cycles share for this symbol at all. So there were three live numbers — 19.90,
9.80, and nothing — and the spread between my own two is **2.06×** from the
inlining boundary alone. Rule 8 earned its place again.

**The symbol is not a fixed quantity.** PGO folds `Worker::do_move`,
`make_move_with`, the TT `probe`, `gives_check`, `legal`, `see_ge` and the
history updates into it; in the non-PGO build those are separate symbols. Any
future citation must name its build.

### 0.2 ★ The decomposition: 72% of the symbol is inlined callees, and it ABSORBS NOTHING NET

The commission's hypothesis was that `search::<NonPv>` is "a mix of its own work
and stall it inherited from everything it calls". **Measured, that is false at
the symbol level and true only at the line level.** The two must be separated,
and no previous study did:

**Layer 1 — TEXTUAL (what code is in the symbol).** DWARF inline frames on a
cycles profile of `pgo52`, `tools/perf52/inline52.py`:

| phase | % of the symbol's cycles |
|---|---:|
| **`do_move`** (make_move + accumulator push) | **29.12 %** |
| **search's OWN code** | **27.29 %** |
| `next_move` (move picker) | 7.71 % |
| `legal` | 5.29 % |
| `undo_move` | 3.75 % |
| `all_threats` | 3.69 % |
| `update_all_stats` | 3.16 % |
| `see_ge` | 2.77 % |
| `gives_check` | 2.03 % |
| `probe` (TT) | 1.79 % |
| `reduction` | 1.73 % |
| `stat_score` | 1.62 % |
| `correction_value` | 1.56 % |

**Only 27.29 % of the largest symbol in the engine is search's own code** —
study 33 measured 32.87 % of *stall*; this is the first measurement on cycles.

**Layer 2 — ATTRIBUTION (whose cost the cycles are).** AMD IBS, which tags a
micro-op at issue and records that op's own rip, so there is no skid by
construction. Six runs, **1,512,903 tagged ops** on `pgo52`:

| `search::<NonPv>`, `pgo52` | value |
|---|---:|
| share of engine **cycles** (sampled, skiddy) | 25.38 % |
| **share of engine micro-OPS (IBS, precise)** | **28.46 %** |
| share of engine loads | 26.64 % |
| share of engine L1 misses | 11.85 % |
| **share of engine load-miss LATENCY** | **8.50 %** |

**It spends fewer cycles than its share of executed ops — 0.89× the engine's
mean cycles-per-op.** The non-PGO build says the same thing harder: 12.33 % of
cycles on **15.61 %** of ops, a ratio of **0.79**. A symbol that absorbed its
callees' stall would sit *above* 1.0. `search::<NonPv>` is, by this measure,
**the most op-efficient large symbol in the engine**, and 91 % of its loads hit
L1 (study 33 §5.1, reconfirmed).

**So the honest decomposition of the site is:**

> It is big because **it executes 28.46 % of the engine's micro-ops**, of which
> roughly 72 % belong to callees the compiler inlined into it. It is not a stall
> site, it absorbs nothing net, and there is no hidden memory cost inside it to
> recover. **Its cost is its op count, and that is the only currency that can
> move it.**

### 0.3 ★ …and at the LINE level the cycles counter is wrong by 46×

The commission was right that something is mis-charged — just not at the
boundary it expected. Inside the symbol, the two instruments disagree violently:

| inlined leaf | % of symbol's **cycles** | % of symbol's **ops** (IBS) | ratio |
|---|---:|---:|---:|
| **`push_sota_dirty`** (`src/eval/mod.rs:372`) | **11.59 %** | **0.253 %** | **46×** |

**The #1 hot line in the biggest symbol in ARCHON executes a quarter of one
percent of its instructions.** It is the accumulator's 16-byte `Meta` record
store, and it is where the re-order buffer's head parks while the *previous*
node's outstanding misses drain. Study 33 §4.1 suspected exactly this ("a
non-precise cycle-counting event names where the ROB head parks, not where the
cost is") on the same line under its old attribution; **this is the measurement,
and it is a 46× error.** It also fully explains Layer 1's apparent anomaly —
`do_move` at 29.12 % of cycles on 20.06 % of ops — without any absorption story:
the parking is inside `do_move`.

**Consequence for every study in this line: `perf annotate` on `cycles:u` or on
`ex_no_retire.load_not_complete` must not be used to rank work inside a symbol
on this box.** `tools/perf52/ibsline52.py` is the replacement.

### 0.4 ★ What LANDED

**A speedup, node-identical, measured 15/15 on three counters** — `rfp52`,
§5.4. Reverse-futility pruning computed the "none of my pieces is threatened"
flag **unconditionally**, and that flag costs a full `compute_threats()` sweep
because RFP runs before any child search, so the board's threat memo is cold
there. It is the single largest *call site* in the symbol: **`all_threats` is
3.17 % of `search::<NonPv>`'s ops = 0.90 % of ALL engine ops.** Because
`RFP_THREAT_MARGIN` is positive, the flag can only ever make pruning easier, so

```
C(flag)  ==  C(false) || (flag && C(true))
```

exactly — and short-circuiting confines the sweep to the 55-wide band where the
relaxation decides.

| | median | se | neg/n | per node |
|---|---:|---:|---:|---:|
| **`instructions:u`** | **−0.593 %** | 0.000 | **15/15** | **−36.5** |
| **`ls_dispatch.ld_dispatch:u`** | **−1.164 %** | 0.074 | **15/15** | **−26.6** |
| **`branches:u`** | **−0.875 %** | 0.000 | **15/15** | **−5.8** |

Node-identical (**5,001,562** / **1,739,588**), null 0.000 % / +0.057 % /
0.000 %. **No cycles claim** — §5.4 says why, and says that cycles' median
points the other way. For scale: study 35 landed −25.0 i/n, study 36 −19.5 i/n.

Also landed:

1. **`tools/perf52/ibsline52.py` — a precise per-line / per-inline-frame /
   per-operand-class census inside one symbol.** The repo had per-symbol IBS
   (study 49) and per-region IBS (studies 50/51); this is the per-*instruction*
   view, and it is what showed the 46× error in §0.3, priced every arm before it
   was built, and found the call site above. Plus `inline52.py`,
   `buildpgo52.sh` and the study-49 panel harness rebased on this frame.
2. **One arm built, gated, measured and REFUTED — with its cause isolated by a
   control** (§5.1, §5.2): a range hint that deleted provably-dead divide guards
   and was paid for in spills.
3. **The mechanism that refutation exposed — `search::<NonPv>` is
   ALLOCATOR-FRAGILE** — and the standing rule it yields, **which then predicted
   the sign of both arms**: delete a computation (`rfp52`, −26.6 loads/node),
   never add a value (`div52`, +19.1 loads/node).
4. **A full price list for every micro-lever class in the symbol** (§5.3).
5. **Three levers priced and closed** (§0.5) and **one new axis opened** (§0.6).

### 0.5 Three levers priced and CLOSED, with numbers

Reported prominently because a closed lever is a result:

1. **Integer division — 2.043 % of all engine cycles, and it is NOT worth
   taking.** `ex_div_busy` = 74.85 cycles/node, `ex_div_count` = 9.713 divide
   ops/node, **7.71 cycles per divide** (first measurement of this in the
   repo). But **40.49 % of all divide ops are in `quantise_halfka`, which is
   STARTUP**, and `search::<NonPv>` holds 19.71 % — i.e. **1.914 divides/node,
   ≈14.8 cycles/node, 0.40 % of frame** as an absolute ceiling for removing
   *every* divide in the search node. Its four `idiv` sites are
   `mod.rs:4310` (`reduction`), `mod.rs:3386` (`lmr_depth`), `mod.rs:3648` and
   `mod.rs:4050`; IBS says the first two carry essentially all of it (0.179 %
   and 0.174 % of the symbol's ops) and the last is never executed. A correct
   signed magic-reciprocal rewrite is fiddly, and 0.40 % of frame cannot be
   adjudicated on `cycles:u` on this box. **Priced, declined, documented.**
2. **The TT probe's memory cost is already at its floor.** `probe` is
   **1.92 % of the search node's ops but 44.72 % of its load-miss latency** —
   the densest site in the engine (326.1 cycles/miss, 34.3 % miss rate, 951 of
   3,143 sampled loads served from DRAM). The obvious fix does not exist:
   `src/tt.rs:326-335` already lays `Cluster` out as **32 bytes, `align(32)`**,
   so a probe touches exactly one cache line, and a `const` assertion pins it.
   Study 42 already spent the prefetch placements.
3. **"One hot spill slot" — refuted, my own hypothesis.** A static count says
   **50.5 %** of the symbol's load instructions are `%rsp`-relative, which
   looked like a concentrated register-pressure defect. Dynamically it is
   **29.0 % of its loads, and 19.3 % of all its ops, spread over 117 stack
   slots with the hottest at 0.87 %** — top 8 slots are only 26.6 % of the
   spill traffic. There is no single variable to fix. **The static count
   overstated the case and is not a usable guide.**

### 0.6 The one live forward pointer, measured but UNPRICED

**Nobody has ever measured store-to-load forwarding in this engine.**
`search::<NonPv>` is the **#1 symbol at 32.32 % of all engine
`ls_bad_status2.stli_other` events** (non-forwardable conflict with an older
store) — above its 28.46 % op share, and the engine runs **59.7 failed
forwards/node against 111.7 successful ones (35 % failure rate)**. This is the
natural mechanism for a site with 19.3 % of its ops in spill traffic and a
46× cycles/ops parking anomaly, and it is a different axis from every counter
this line of studies has used. **It is reported as an observation, not a
result:** I did not establish the per-event cost, did not run a null on the
counter, and a naive 12-cycle penalty would price it at ~20 % of frame, which
is certainly too high because the penalty overlaps. **Do not build on this
number until it has a null and a known-answer probe.**

---

## 1. THE FRAME, RE-DERIVED (rule 8)

`tools/perf52/census52.sh`, 3 unmultiplexed groups × 3 reps, median, on an
**idle** box (all 8 physical cores 0.0 % before starting).

| | study 49 (`base49`) | study 50/51 (`base50/51`) | **this study (`base52`)** |
|---|---:|---:|---:|
| **`instructions:u`** | 6153.06 | 6153.10 | **6153.06** |
| `cycles:u` | 3753.76 | 3720.81 / 3675.33 | **3665.37** |
| `ls_dispatch.ld_dispatch:u` | 2275.86 | 2284.66 / 2282.64 | **2278.75** |
| `ex_no_retire.load_not_complete:u` | 1764.05 | — | **1685.35** |
| `ls_any_fills_from_sys.all:u` | 575.74 | — / 566.41 | **564.60** |
| `ls_alloc_mab_count:u` | 22354.28 | — | **21595.16** |

**`instructions:u` reproduces study 49's 6153.06 to the last printed digit** —
raw total 30,774,892,792 — which is both a build check and a statement that
studies 50 and 51 landed nothing that moved the instruction count.

**Every percentage-of-cycles figure in this document is against my own measured
3665.37**, never an inherited one. The commission's 4162.8–4179.7 frame is from
a different (busier) night; study 36's 8262.9 is wrong by ~2× and is not used.

### 1.1 The box stopped being quiet at 04:00, and the boundary is stated

Every number in §0 and §2–§4 was recorded between 03:50 and 04:00 with all
cores at 0.0 %. At 04:04 the D49 trainer resumed (152 % CPU) **and a 12-process
`fastchess` screen started** (load average 8.31). The tell was immediate and is
recorded rather than smoothed: an identical `bench` invocation that had read
**18.33 G cycles** read **24.33 G**, +33 %.

**Consequences, applied throughout:**

* **No `cycles:u` panel is run and no cycles claim is made for any arm.** The
  commission already put the byte-identical null at +0.517 % [−2.71, +1.78];
  under this load it is far worse.
* **`instructions:u` is unaffected** — it counts *this process's* retired
  instructions and is deterministic — and it is therefore the adjudicating
  counter for every arm in §5.
* Two contaminated runs are reported in §7 with their contamination, not
  deleted.

---

## 2. THE INSTRUMENT, AND THE TWO BUGS IT CAUGHT IN ME

`tools/perf52/ibsline52.py` is new: a **precise per-line, per-inline-frame,
per-operand-class census inside one symbol**, joining IBS `IbsOpRip` samples
against the symbol's own disassembly and DWARF inline stacks. Previous studies
had per-symbol IBS (study 49) and per-region IBS (studies 50/51); this is the
per-*instruction* view inside a symbol, which is what a decomposition needs.

**Two bugs it caught, both of the class that silently produces a headline:**

1. **`qsearch::<NonPv>` contains `search::<NonPv>` as a substring.** My first
   engine-relative share summed both and read **13.02 %** instead of 11.54 %.
   Caught because it disagreed with `perf report`'s own row by exactly
   `qsearch`'s 1.48 %.
2. **`nm` reports the v0-mangled name; `perf script` demangles.** The join
   returned **zero samples in the symbol** and every downstream share was
   0.00 %. A silent-zero failure mode, not a crash — the tool now takes the
   perf-side display name as a separate argument and the docstring says why.

**A third caveat, found and not worked around:** IBS samples micro-ops, so
**microcoded instructions are over-represented relative to `instructions:u`**.
The four `idiv` sites read 0.359 % of the symbol's ops by IBS but
`ex_div_count` says 1.914 divides/node, a **3.3× disagreement** in the
direction IBS would be expected to err. Single-uop loads and stores are
unaffected, so §3's operand-class census stands; **do not read IBS op shares as
instruction counts for complex instructions.**

---

## 3. WHAT THE SYMBOL EXECUTES

`pgo52`, 404,744 IBS-tagged ops inside the symbol (26.99 % of 1,499,713
engine-wide), 5,452 of its 8,821 instructions hit at least once.

### 3.1 Operand class — static shape vs what actually runs

| class | static instructions | static % | **dynamic ops** | **dynamic %** |
|---|---:|---:|---:|---:|
| `load-rsp` (spill reload) | 1145 | 13.0 % | 43,031 | **10.6 %** |
| `load-mem` (real data) | 1362 | 15.4 % | 104,781 | **25.9 %** |
| `store-rsp` (spill) | 335 | 3.8 % | 28,060 | **6.9 %** |
| `store-mem` | 194 | 2.2 % | 11,456 | 2.8 % |
| `reg-only` | 5785 | 65.6 % | 217,416 | 53.7 % |

**Loads are 36.5 % of the symbol's executed ops; 29.0 % of those loads are
spill reloads.** Total stack traffic (reload + spill store) is **19.3 % of the
symbol's ops ≈ 5.5 % of all engine ops**. That is a real aggregate cost with,
per §0.5 item 3, no concentrated handle: 117 distinct slots, hottest 0.87 %.

### 3.2 Phase, on the precise instrument

Side by side with Layer 1's cycles view — the divergence is the whole point:

| phase | % cycles | **% ops** | % loads | % miss-latency |
|---|---:|---:|---:|---:|
| search's own code | 27.29 | **32.36** | 33.37 | 13.95 |
| `do_move` | 29.12 | **20.06** | 23.68 | 10.73 |
| `next_move` | 7.71 | 6.17 | 5.78 | 2.64 |
| `legal` | 5.29 | 4.74 | 4.60 | 3.17 |
| `update_all_stats` | 3.16 | 3.83 | 1.97 | 0.39 |
| `all_threats` | 3.69 | 3.17 | 3.36 | 2.51 |
| `undo_move` | 3.75 | 3.11 | 4.04 | 5.33 |
| `see_ge` | 2.77 | 2.66 | 3.38 | 3.55 |
| `correction_value` | 1.56 | 2.61 | 2.64 | 5.96 |
| `reduction` | 1.73 | 2.58 | 1.58 | 0.57 |
| `gives_check` | 2.03 | 2.56 | 2.73 | 0.07 |
| `stat_score` | 1.62 | 2.00 | 2.74 | 0.56 |
| **`probe` (TT)** | 1.79 | **1.92** | 1.72 | **44.72** |

Two rows carry the study. **`do_move` is 1.45× cycles-per-op** — that is §0.3's
parking, not a mechanism. **`probe` is 23× miss-latency-per-op** — that is real,
and §0.5 item 2 says it is already at its floor.

### 3.3 The op distribution is FLAT — there is no hotspot to attack

Top source lines by precise ops, inside the symbol:

| ops % | line |
|---:|---|
| 6.51 % | `src/search/mod.rs:?` (no line info) |
| 4.38 % | `src/board.rs:?` |
| 1.95 % | `src/bitboard.rs:187` |
| 1.70 % | `src/board.rs:826` |
| 1.46 % | `src/moves.rs:15` |
| 1.44 % | `src/movepick.rs:736` |
| 1.32 % | `src/moves.rs:62` |
| **1.20 %** | **`src/search/mod.rs:4310`** (`reduction`'s divide) |
| 1.18 % | `src/moves.rs:66` |

**The hottest line with real debug info is 1.95 %.** Study 33 called the site
"distributed"; this confirms it on a precise instrument and quantifies it. It is
also why §0.5 closes three levers rather than opening one: a flat distribution
means the wins are many-and-small, and the only counter that can see a
many-and-small win on this box is `instructions:u`.

### 3.4 Cross-check: the non-PGO symbols PGO folds in

`base52` (non-PGO), startup excluded: `search::<NonPv>` is **15.70 %** of engine
ops. Adding the symbols PGO inlines into it — `make_move_with` 3.44 %,
`update_continuation_histories` 2.25 %, `Worker::do_move` 1.94 %, `see_ge`
1.80 %, `refresh_check_info` 1.74 %, `correction_value` 1.42 %, `unmake_move`
1.25 %, `ensure_threats` 1.20 %, `gives_check` 0.92 %, `probe` 0.81 %,
`attackers_to_exist` 0.71 %, `stat_score` 0.67 %, others — sums to **34.35 %**
against the PGO symbol's **28.73 %**.

**The sum deliberately OVERSHOOTS and that is the correct behaviour**: those
callees also serve `qsearch::<NonPv>` (3.02 % of ops) and `search::<Pv>`, whose
calls do not fold into this symbol. The check is that it brackets the PGO figure
from above, which it does; it is not an identity and is not presented as one.

---

## 4. WHAT THIS FORBIDS AND WHAT IT LICENSES

**Forbidden at this site, now with two independent measurements each:**

* **Any locality or prefetch intervention.** 8.50 % of engine miss latency on
  28.46 % of its ops; 91 % L1 hit rate. Study 31 measured this as a null arm,
  study 33 §5.3 gave it a reason, and §3.2 above reproduces it per-phase. The
  one dense sub-site (`probe`, 44.72 % of the symbol's miss latency) is already
  at its architectural floor (§0.5 item 2).
* **Reading work off a `cycles:u` or `load_not_complete` annotation inside the
  symbol.** §0.3: wrong by 46× on the hottest line.
* **Cold-code outlining** — study 41 measured 83.9 % of the symbol's bytes as
  executed and `hcs41` as +1.485 % L1i misses.

**Licensed, and it is the only thing licensed: reduce the NUMBER of ops the
node executes.** At 28.46 % of engine ops and a cycles/op ratio of 0.89, the
payoff is close to linear and `instructions:u` measures it exactly (byte-
identical null 0.000 %, se 0.000) even on a loaded box.

---

## 5. ARMS

### 5.1 ★ REFUTED, and the refutation is the study's most useful finding — `div52`

**The hypothesis, and it was measured before it was built.** `reduction`
(`src/search/mod.rs:4304`) divides by `self.root_delta`, a plain `i32`. LLVM
cannot prove a plain `i32` divisor is non-zero or is not `-1`, so it emits a
divide-by-zero test **and** an `INT_MIN / -1` overflow test before the `idiv` —
on every call, once per move in the move loop, to protect against values that
cannot occur (`root_delta` is `beta - alpha` at the root and the function
already `debug_assert!`s it positive). Read off the shipped disassembly at
`0x528ed`:

```
mov  0x66d8(%rbx),%ecx     ; self.root_delta
test %ecx,%ecx
je   <panic_const_div_by_zero>
mov  %ecx,%edx
lea  -0x80000000(%rbp),%eax
not  %edx
or   %eax,%edx
je   <panic_const_div_overflow>
...
idiv %ecx
```

Priced on the precise instrument **before building**: the guard block is
**0.654 % of `search::<NonPv>`'s executed micro-ops**, against **0.179 %** for
the `idiv` it guards. **The guards cost 3.7× the division.** (The engine's other
hot divide, `mod.rs:3386`, is unguarded — LLVM proved the `LMR_DIVISOR` const
table's entries safe — so this was the only such site worth touching.)

**§2's IBS caveat cuts in this claim's favour, not against it**, which is worth
stating because it looks like a threat to it: IBS samples micro-ops, so the
microcoded `idiv` is *over*-represented (the four sites read 3.3× high against
`ex_div_count`) while the guard block is all single-uop `test`/`je`/`not`/`or`.
The true instruction ratio is therefore **larger** than 3.7×.

**The arm.** One line, `let root_delta = self.root_delta.max(1);` — a range
hint, not a clamp, narrowing the divisor to `[1, i32::MAX]` and discharging both
proofs.

**It did exactly what it was designed to do, and it still LOST.** 15 rounds,
blocked Latin square, byte-identical null riding in the rotation:

| `div52` vs `base52` | median | trimmed | se | neg/n | p |
|---|---:|---:|---:|---:|---:|
| **`branches:u`** | **−0.452 %** | −0.452 | **0.000** | **15/15** | 0.0001 |
| **`instructions:u`** | **+0.052 %** | +0.052 | **0.000** | **0/15** | 0.0001 |
| **`ls_dispatch.ld_dispatch:u`** | **+0.836 %** | +0.813 | 0.059 | **0/15** | 0.0001 |
| `cycles:u` | +0.327 % | +0.345 | 0.270 | 6/15 | 0.61 |

**The null, published as required:** `branches:u` **0.000 %**, `instructions:u`
**−0.000 %**, `ld_dispatch` +0.032 %, `cycles:u` +0.085 %. Realised mean panel
position **1.00** for all three labels.

The branch result is a *confirmation*, not a consolation: −0.452 % of 663.70
branches/node is **−3.00 branches/node**, which is exactly two guard branches ×
1.5 `reduction` calls/node. The mechanism fired. **And the site charged more for
it than it was worth**: +3.2 instructions/node and **+19.1 loads/node**, in the
one currency study 33 §5.3 explicitly licensed ("reducing the NUMBER of loads is
licensed, the payoff is linear").

### 5.2 ★ WHY it lost — `search::<NonPv>` is ALLOCATOR-FRAGILE, and the control isolates it

Static diff of the two binaries' symbol:

| | `base52` | `div52` | Δ |
|---|---:|---:|---:|
| symbol bytes | 16,089 | 16,231 | +142 |
| instructions | 3,383 | 3,397 | +14 |
| **`%rsp` load sites** | **485** | **516** | **+31** |
| **stack frame** | **2,824 B** | **2,840 B** | **+16** |

**Deleting ~7 instructions of provably-dead guard code bought 31 extra static
spill reloads.** Freeing the divisor's live range let LLVM re-run allocation on
a 3,383-instruction function whose frame is already 2,824 bytes and 19.3 % of
whose ops are already spill traffic (§3.1) — and it came out worse.

**The obvious objection is "you perturbed a fragile function, any edit would do
that", and a control refutes it.** I rebuilt the same line as an inert
`let root_delta = self.root_delta;` — same new binding, same shape, no range
hint. It compiled **byte-identical to `base52`** (`.text` md5 `b4f62057` on
both). So a new binding is free; **the entire 31-spill regression is
attributable to the range hint specifically**, i.e. to the guard removal, and
not to having touched the function.

**This is the generalisable result, and it retro-explains three studies.** Study
31 measured a null arm here, study 33 landed no speedup here, and this study's
own best-priced micro-optimisation went backwards — all at a site whose
published diagnosis was "average per load, just reduce its op count". The
missing term is that **`search::<NonPv>` sits at a register-allocation local
optimum that punishes local edits**: the payoff of any change inside it is
(instructions removed) − (spill traffic LLVM adds back), and the second term is
not small and not predictable from the source diff.

**The standing rule this study adds:** at this site, an arm must **delete a live
value or delete a whole computation**, not add a hint, a clamp or a temporary.
Anything that merely rearranges what is live across the move loop should be
expected to be paid for in spills, and **must be measured on
`ls_dispatch.ld_dispatch` as well as on `instructions:u`** — this arm looked
like a win on branches alone.

**Reverted.** The tree is back to `base52` byte-for-byte (`chk52` rebuilt from
the reverted source, `.text` md5 `b4f62057`).

### 5.3 ★ THE CENSUS THAT REPLACES "attack what the decomposition says is there"

Rather than assert that the site is flat, I **priced every candidate class
inside it** on the precise instrument. All four, together, in one table — the
symbol is 28.46 % of engine ops, so the right-hand column is what any arm in
that class could ever be worth:

| candidate class inside `search::<NonPv>` | % of the SYMBOL's ops | **% of ALL engine ops** |
|---|---:|---:|
| **register SPILL traffic** (`%rsp` loads + stores) | **19.3 %** | **5.49 %** |
| all **110** bounds-check branches | 1.714 % | 0.488 % |
| the four `idiv`s **plus** their guard blocks | 0.833 % | 0.237 % |
| the single hottest source line with debug info | 1.95 % | 0.555 % |

**Spill traffic is 11× larger than every micro-lever in the symbol combined.**
That is the answer to the commission's part 2, and it is a measurement, not an
impression: the bounds checks are 110 sites averaging **0.016 %** each, hottest
**0.095 %**; the divides are two live sites; the source-line distribution tops
out at 1.95 %.

**But read what this table is and is not.** It prices *classes* — instruction
kinds spread over the whole symbol — and it correctly says there is no
worthwhile arm in any of them. **It does not price individual call sites, and
that is where the study's landed win came from**: `all_threats` is one call site
at 0.90 % of engine ops (§3.2's phase table, not this one), nearly twice the
entire bounds-check class, and §5.4 removed most of it. **The lesson is that a
class census can be flat while a site census is not**, and both were needed —
had I stopped at this table I would have concluded the symbol was finished, and
I would have been wrong by −36.5 instructions/node.

What the table does establish is where the *remaining* bulk is: the cost of
keeping a 3,383-instruction function's state alive in 16 architectural
registers.

### 5.4 ★ LANDED — `rfp52`: the RFP threat map becomes lazy (−36.5 instructions, −26.6 loads, −5.8 branches per node)

**Where it came from.** §5.3's census says the symbol has no algorithmic
redundancy *of consequence* left — but "of consequence" was measured over
classes (spills, bounds checks, divides), not over individual call sites. The
phase table (§3.2) has one row that is a **single call site** rather than a
class: **`all_threats` at 3.17 % of the symbol's ops = 0.90 % of ALL engine
ops.**

**The site.** Reverse-futility pruning relaxes its margin when none of our
pieces is threatened (Wave-G item 23, D30, Reckless R8), and it computed that
flag *unconditionally* before testing the margin:

```rust
let no_own_threat = (self.board.all_threats() & self.board.occupancy(us)).is_empty();
let margin = rfp_prune_margin(depth, …, no_own_threat);
if eval - margin >= beta { return …; }
```

`Board::ensure_threats` memoises on the position key, so this looks free — but
**RFP runs before any child search, so the entry is cold and this is a full
`compute_threats()` sweep on every node reaching the RFP gate.**

**Why the deferral is EXACTLY equivalent, and it is an ordering argument not an
approximation.** `rfp_prune_margin` subtracts `RFP_THREAT_MARGIN * flag`, and
`RFP_THREAT_MARGIN` is **+55** (`params.rs:349`), so `m(true) ≤ m(false)`
unconditionally — the flag can only make pruning *easier*. Writing `C(f)` for
`eval - m(f) >= beta`, that ordering gives `C(false) ⇒ C(true)`, hence

```
C(flag)  ==  C(false) || (flag && C(true))
```

for **both** values of `flag`, and Rust's `||`/`&&` short-circuit confines
`all_threats()` to the band where the relaxation actually decides. The
equivalence survives the EXP-UNC-1 rescale because that scale is non-negative
and `as Value` truncation is monotone, so it cannot reorder the two margins.

**Node identity is the proof, and it is exact:** `5,001,562` and `1,739,588`.
A predicate that was not equivalent would move the tree, and this one does not.

**Measured.** 15 rounds, blocked Latin square, byte-identical null riding in the
rotation, realised mean position **1.00** for all three labels:

| `rfp52` vs `base52` | median | trimmed | se | neg/n | p | per node |
|---|---:|---:|---:|---:|---:|---:|
| **`instructions:u`** | **−0.593 %** | −0.593 | **0.000** | **15/15** | 0.0001 | **−36.5** |
| **`ls_dispatch.ld_dispatch:u`** | **−1.164 %** | −1.189 | 0.074 | **15/15** | 0.0001 | **−26.6** |
| **`branches:u`** | **−0.875 %** | −0.875 | **0.000** | **15/15** | 0.0001 | **−5.8** |
| `cycles:u` | +0.595 % | +0.297 | 0.267 | 6/15 | 0.61 | — |

**The null:** `instructions:u` **0.000 %**, `branches:u` **0.000 %**,
`ld_dispatch` +0.057 % (se 0.044), `cycles:u` −0.050 %.

**Every counter that can resolve this moves the right way, 15/15, and the
largest move is in loads — the one currency study 33 §5.3 licensed.** For scale:
study 35's landed `after` was −25.0 i/n and study 36's landed `flathist`
−19.5 i/n; this is **−36.5 i/n and −26.6 loads/n**.

**No cycles claim is made, and the honest reading is that cycles points the
other way**: the median is +0.595 % with se 0.267, against a null that read
−0.050 %, +0.085 %, +0.233 % and +0.884 % across tonight's four panels. The
counter's spread is larger than the effect and its sign is not stable across
nulls, so it neither confirms nor contradicts the arm — it simply cannot see it.
**Anyone re-running this on a quiet box should re-null first and report the
cycles result whichever way it falls.** Elo is not estimated: −0.6 % of
instructions is not −0.6 % of time, and this study has no measurement that
converts one to the other.

**Contrast with §5.1, because it is the same rule being obeyed rather than
broken.** `div52` *added* a value (a range hint) and paid +19.1 loads/node.
`rfp52` *deletes a computation* and collects −26.6 loads/node. The
allocator-fragility rule predicted the sign of both.

### 5.5 The natural experiment on the spill block, and why it was NOT run

The one lever that acts on 5.49 % of engine ops is the **inlining boundary**:
`do_move` is 20.06 % of the symbol's ops and PGO folds it in, so
`#[inline(never)]` would delete a whole register set from the move loop — which
is exactly the shape §5.2's rule endorses.

**I measured the trade before building it, from the two binaries I already
had**, and that is what stopped me. Same source, same frame, same node count:

| | `base52` (do_move NOT inlined) | `pgo52` (do_move inlined) | Δ |
|---|---:|---:|---:|
| `instructions:u` | 6153.06 | 6046.81 | **−1.73 %** |
| `branches:u` | 663.69 | 623.41 | **−6.07 %** |
| **`ls_dispatch.ld_dispatch:u`** | 2281.82 | 2301.64 | **+0.87 %** |

**Inlining `do_move` buys −106 instructions and −40 branches per node for
+19.8 loads per node** — the same trade, in the same direction and to within
4 % the same magnitude, as `div52`'s +19.1 loads/node. So an `inline(never)`
arm would move `instructions:u` **up** and `ld_dispatch` **down**, and the only
counter that could break that tie is `cycles:u`, whose null tonight is
+0.23 % to +0.88 % (se 0.39–0.47) against an effect of order 0.3 %.

**The experiment is not unpromising; it is UNADJUDICABLE on this box tonight,
and running it would have produced a number I could not interpret.** It is
written up as the next study's first arm in §9 with the conditions it needs.
(The base52↔pgo52 pair is a *confounded* comparison — two different compiler
pipelines, not one attribute — so it is quoted as the reason for a decision,
never as the result.)

---

## 6. GATES

| gate | result |
|---|---|
| **`CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh`** | **All checks passed** |
| wl50 eval hash | **`2e1ede95…79c4`** — exact |
| d13 node total (gate) | **1,739,588** — exact |
| `Bench:` trailer (depth-6 default build) | **587683** — unmoved |
| node identity, study frame — `base52`, `null52`, `div52`, `bind52`, `chk52`, `rfp52`, `rfp52b` | **5,001,562** exact on every binary |
| d13 node identity — `base52`, `div52`, `rfp52`, `rfp52b` | **1,739,588** exact |
| `rfp52b` (clippy-cleaned source) vs `rfp52` (measured binary) | `.text` **byte-identical**, md5 `e160e553` |
| `instructions:u`/node vs studies 49/50/51 | **6153.06** vs 6153.06 / 6153.10 / 6153.10 |
| `ls_any_fills_from_sys.all` vs study 51 | 564.60 vs 566.41 (−0.32 %) |
| byte-identical null, `instructions:u` | **0.000 %**, se 0.000 |
| byte-identical null, `branches:u` | **0.000 %**, se 0.000 |
| byte-identical null, `cycles:u` | +0.085 % / +0.233 % / +0.884 % — **not adjudicating** |
| realised mean panel position | **1.00** for every label in every panel |
| revert reproduces base | `chk52` `.text` md5 `b4f62057` = `base52` |

**No pin moves, and none should.** `rfp52` is exactly behaviour-preserving, so
all three pins — `587683`, `2e1ede95…79c4`, `1739588` — must reproduce
unchanged, and they do. **A moved pin here would have been evidence the
predicate restructure was not the identity §5.4 proves it is**, which is
precisely why the gate is the right check for this arm and why no game is owed
(§9.1).

An earlier gate run was ABORTED and is not quoted: it was launched against the
clean base and I edited `src/search/mod.rs` while it was in flight, so a test
binary vanished mid-run under the concurrent `cargo` build and it failed with
`No such file or directory`. **That was my process error, not a real failure**;
the gate above was re-run serialised, with no other build touching the tree.

D24 perft ritual not required: nothing under `movegen`, `board` mutation,
`zobrist` or `attacks` is touched.

---

## 7. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **The store-to-load-forwarding lead of §0.6 is unpriced.** 32.32 % symbol
   share and 59.7 failures/node are real counter readings, but I ran no null on
   `ls_bad_status2.stli_other`, established no per-event cost, and the two runs
   that produced the per-node rate were taken **after** the box got busy (§1.1).
   The share is from a clean `perf record` but its skid is uncharacterised.
   **Nothing in this study depends on it.**
2. **No cycles claim is made for anything.** The box went from idle to load
   average 8.31 mid-study and an identical run moved +33 %. The
   division-removal ceiling of 0.40 % of frame (§0.5 item 1) is therefore an
   arithmetic estimate from `ex_div_busy`, never a measured speedup, and
   "divider busy" is not the same as "cycles saved if removed" — the divider
   runs in parallel with other execution, so 0.40 % is an upper bound and
   probably a loose one.
3. **The PGO figure rests on ONE PGO build with my own training workload**, the
   same caveat study 33 §10 item 2 raised. My `pgo52` measured only **+2.3 %
   NPS** over `base52`, well under the +8.1 % the project standard reports for
   `tools/build_pgo.sh`, because `--profile profiling` is not the shipping
   profile. The 25.38 % is real for that binary; it is not automatically the
   shipping binary's number.
4. **I did not reproduce study 49's `stall/own-miss-latency` column** for this
   symbol on my own build. The commission asked for it; the counter it needs
   (`ex_no_retire.load_not_complete`) reads +3.478 % on a byte-identical null at
   n=6 by the commission's own measurement, and the box went busy before I could
   run it clean. **I answered the same question with a better instrument**
   (cycles-share vs precise-op-share, §0.2), and that substitution is a
   deviation from the commission stated here rather than buried.
5. **Layer 1 and Layer 2 use different denominators by construction** — Layer 1
   is a share of the symbol's *cycles*, Layer 2 a share of its *ops*. They are
   presented side by side in §3.2 precisely so the divergence is visible, but no
   single "% of the 25.38 % that is absorbed" scalar is quoted, because the two
   senses of "absorbed" (textual inlining vs out-of-order parking) do not
   compose into one number and forcing them to would be the beautiful-story
   failure mode.
6. **`src/search/mod.rs:?` is 6.51 % of the symbol's ops with no line
   attribution**, and `src/board.rs:?` another 4.38 %. Almost 11 % of the
   decomposition is therefore unlocalised within its file. This is a limit of
   `line-tables-only` DWARF through heavy LTO inlining, not a bug I could fix.
7. **The allocator-fragility finding rests on ONE arm plus ONE control.** The
   control is strong — the inert binding is byte-identical, so the +31 spills
   are attributable to the range hint and not to the edit — but "this symbol
   punishes local edits" is a generalisation from a single instance of it. The
   `base52`↔`pgo52` trade in §5.5 points the same way at the same magnitude
   (+19.8 vs +19.1 loads/node), which is *suggestive corroboration and nothing
   more*: those are two different compiler pipelines, so the agreement could be
   coincidence. **Two data points that agree to 4 % is exactly the kind of tidy
   result rule 6 says to attack, and I could not attack it further tonight.**
   Anyone relying on the rule should re-test it with a third, unrelated edit.
8. **I did not establish that removing the `div52` guards is worthless in
   principle** — only that *this formulation* of it loses. A formulation that
   discharges the proofs without creating a new SSA value (e.g. carrying the
   positivity in `root_delta`'s **type**) might not pay the spill. `NonZeroI32`
   is not that type: its validity range excludes 0 only, so it discharges the
   divide-by-zero proof and leaves the `INT_MIN / -1` test standing, and the
   standard library has no positive-integer type. I judged a ~0.15 % target not
   worth further builds in a symbol just measured to be allocator-fragile;
   that is a judgement call and it is recorded as one.
9. **The 42 bounds checks were priced but none were attacked.** §5.3 prices the
   whole class at 0.488 % of engine ops across 110 branch sites averaging
   0.016 % each. That is a reason not to spend builds, not a proof that a
   bulk removal is impossible.

---

## 8. REPRODUCTION

```bash
# 0. worktree needs the gitignored net visible under its own path
mkdir -p data/nnue/nets/screens
ln -sfn /path/to/d48x-1280-sb520-ema-s600.cenn \
        data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn

# 1. the two builds, and the node gate BEFORE any timing
bash tools/perf52/build52.sh base52          # non-PGO release
bash tools/perf52/buildpgo52.sh pgo52        # PGO + DWARF line tables
cp bin52/base52 bin52/null52                 # the byte-identical null
./bin52/base52 bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn               # must print 5001562

# 2. the frame (rule 8) -- on an IDLE box; check all 8 physical cores first
bash tools/perf52/census52.sh 3 base52

# 3. the commissioned number, on BOTH builds -- this is the point
for b in base52 pgo52; do
  taskset -c 2 perf record -q -e cycles:u -c 200003 -o cyc-$b.data -- \
    ./bin52/$b bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn >/dev/null 2>&1
  perf report -i cyc-$b.data --stdio -F overhead,symbol --percent-limit 0
done   # exclude load_v2_bytes/quantise_halfka/init_magics from the denominator

# 4. Layer 1 -- textual decomposition by DWARF inline frame
python3 tools/perf52/inline52.py cyc-pgo52.data bin52/pgo52

# 5. Layer 2 -- the precise instrument. `swfilt=1` is what makes IBS work
#    at perf_event_paranoid=2.  Do NOT park the dump on /tmp (tmpfs).
for i in 1 2 3 4 5 6; do
  taskset -c 2 perf record -q -e ibs_op/cnt_ctl=1,swfilt=1/ -c 8000 -d -W \
    -o ~/.cache/perf52/ibs-$i.data -- ./bin52/pgo52 bench 16 1 100000 \
    tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn >/dev/null 2>&1
  perf script -i ~/.cache/perf52/ibs-$i.data \
      -F ip,sym,symoff,weight,data_src >> ~/.cache/perf52/ibs-pgo52.txt
done
python3 tools/perf52/ibs52.py     < ~/.cache/perf52/ibs-pgo52.txt   # per symbol
python3 tools/perf52/ibsline52.py   ~/.cache/perf52/ibs-pgo52.txt bin52/pgo52

# 6. the divider, and why 40% of it is startup
taskset -c 2 perf stat -e ex_div_busy:u,ex_div_count:u,cycles:u -- ./bin52/base52 ...
taskset -c 2 perf record -q -e ex_div_count:u -c 2003 -o div.data -- ./bin52/base52 ...
perf report -i div.data --stdio -F overhead,symbol

# 7. the panel. NULL FIRST -- run it before any arm and publish it.
P52_EV='instructions:u,branches:u,cycles:u,ls_dispatch.ld_dispatch:u' \
P52_PANEL=docs/research/perf-artifacts-52/panel-div.txt \
  bash tools/perf52/exact52.sh 15 base52 div52 null52
python3 tools/perf52/med52.py docs/research/perf-artifacts-52/panel-div.txt \
  base52 div52 null52
```

---

## 9. GAMES, AND WHAT THE NEXT STUDY SHOULD RUN

### 9.1 There is nothing for games to arbitrate, and that is a result not an excuse

The commission's gate says search changes change the tree, so games arbitrate,
and asks for a ready-to-run screen invocation rather than a long match started
here. **No match is owed, because nothing behavioural landed.** The one arm was
`5,001,562` / `1,739,588` node-identical by construction (a range hint cannot
change a quotient), it was refuted on speed, and it is reverted; the two things
this study leaves in the tree are `tools/perf52/` and this document. A
fixed-node match between `base52` and anything here is a dead heat by
construction, exactly as in study 49 §8 item 6.

**If the §9.2 arm is built and wins, it is still node-identical** —
`#[inline(never)]` cannot change the search tree — so it too is settled by
counters, not games. The screen is only needed if a future arm changes the tree,
and the invocation to use then is:

```bash
bash tools/screen.sh <cand-binary> <base-binary>       # see tools/screen.sh
```

### 9.2 The next study's first arm, and the conditions it needs

**Arm: `#[inline(never)]` on `Worker::do_move`, on a PGO pair.** It is the only
lever that acts on the 5.49 %-of-engine-ops spill block (§5.3), and it is the
right *shape* per §5.2 — it deletes a register set from the move loop rather
than adding a hint.

It needs three things this study did not have:

1. **A quiet box, and a `cycles:u` null under 0.15 %.** The expected effect is
   ~+1.7 % instructions against ~−0.9 % loads; only cycles can adjudicate that,
   and tonight's null was +0.23 % to +0.88 %. **Run the null first and abandon
   the arm if it does not clear** — that is rule 1, and it is why the arm was
   not run tonight.
2. **Both sides built by `buildpgo52.sh`, and the PGO profile noise measured**,
   because each invocation retrains. Build the *same* source twice and panel it
   against itself first; that null has never been taken in this repo and it
   bounds every PGO-pair comparison ever made here.
3. **`ls_dispatch.ld_dispatch` on the panel**, not just instructions — §5.1
   looked like a win on branches alone.

**Second arm, if the first is blocked: price `ls_bad_status2.stli_other`**
(§0.6). `search::<NonPv>` is the engine's #1 symbol for it at 32.32 %, the
engine runs 59.7 failed forwards/node, and **nobody has ever measured store
forwarding in ARCHON**. It needs a known-answer probe in the style of study 49's
`skidprobe` — inject a deliberate narrow-store/wide-load conflict in a leaf and
check the counter attributes it — before any number from it is believed.
