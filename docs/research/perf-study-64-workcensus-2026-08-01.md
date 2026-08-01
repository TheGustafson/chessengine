# Perf study 64 — the work census: 55% of the engine is weight rows, and the only multiplier left is the evaluation count

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`4d41d407`**
(post-study-63 handoff) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Artifacts:**
`docs/research/perf-artifacts-64/` · **Branch:**
`worktree-agent-a23a8cb1793682009`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT,
`-C target-cpu=native`, non-PGO, pinned to core 2. A GPU training run shared the
box throughout; every number here is an **exact integer counter** or an **IBS
op-tagged latency share**, neither of which is a wall-clock statistic, so the
contention affects nothing quoted. **No `cycles:u` and no wall-clock panel
appear in this document** — study 63 established that `cycles:u` is
unresolvable on this box and that `wallclock_panel.py`'s three valid runs are
mutually inconsistent; this study deliberately does not add a fourth.

---

## ★ 0. WHAT LANDED, AND THE ONE-LINE ANSWER

**Landed, gate-green, and provably invisible to the shipped build: two
instruments and the census they produced.** No behavioural change, so the
shipped `.text` is **byte-identical** — `base2`, rebuilt from the final tree,
reproduces `base`'s `.text` md5 **`75c9955e`** and its size **760,085**
exactly (§8).

1. **`foldcount::SEARCH` + 17 hooks in `search()`/`qsearch()`** — the
   **search-side work census**, the first in this series. Every counter in
   `foldcount` before today was charged at fold or eval time, i.e. *after* the
   deferral machinery had already thrown away the plies that never needed an
   accumulator, so none of them could answer the question the brief asked:
   *how many full NNUE evaluations does the search compute that it does not
   need?*
2. **`tools/perf64/foldsrc64.py`** — splits a symbol's IBS tag-to-retire
   latency by **which memory stream the stalled op was reading**, by
   disassembling the binary and classifying every instruction offset. Studies
   29–63 could rank symbols; none could see inside one. This is what produced
   the headline below.

**The one-line answer to the brief:**

> **55.25 % of this engine's L1-miss latency is weight-row loads**, spread over
> four kernels, and **7.175 rows/node is already within 6 % of the minimum the
> lazy machinery can reach** (1.13 fold-plies per evaluation). The row count
> has exactly two multipliers left — **evaluations per node (0.4311)** and
> **bytes per row (1280)** — and both are net-architecture decisions, not
> search or kernel work.

**And the number that prices the first of them:** of 2,163,490 fresh
evaluations, **1,328,601 — 61.41 % — are consumed by a single one-sided
threshold comparison** (a reverse-futility test or a qsearch stand-pat cut) and
by nothing else. Those nodes are leaves, so removing an evaluation there
removes its whole fold as well: **≈ 45 % of the engine's L1-miss latency, at
the free-predictor ceiling** (§5). That is 20× the size of anything else left
on the table, and it is **FUNCTIONAL** — it needs a cheaper *predictor*, and
therefore games, not a speed panel.

**Four measured negatives, three of them closing directions the brief named**
(§3, §6): the TT-cutoff reordering is worth exactly **zero** evaluations
(`main_fresh_then_ttcut = 0`); the deferral machinery has **≈ 6 %** of headroom
in fold count, not the large avoidable population the brief hypothesised;
`at_full` is **not** a full replay and is **1.70 %** of latency, not "the
largest structural difference we have"; and the narrow-row route is priced at
**7.25 % of weight-row bytes** rather than the 25 % the static table suggests.

---

## ★★★ 1. THE INSTRUMENT NOBODY HAD: WHERE THE EVALUATIONS GO

`docs/research/perf-artifacts-64/census-search-fcount2.txt`, whole frame,
node-identical (**5,001,562**) and eval-hash exact (**`2e1ede95…79c4`**):

| counter | count | of what |
|---|---:|---|
| `main_nodes` (search entries past the draw/mate gates) | 3,959,978 | |
| …in check | 429,175 | 10.84 % of main |
| …singular verification (`excluded_move`) | 101,446 | 2.56 % of main |
| `qs_nodes` (qsearch entries past the draw gate) | 1,344,983 | |
| …in check | 139,227 | 10.35 % of qs |
| **`main_eval_ttserved`** | **2,088,047** | **60.65 %** of main evals |
| **`main_eval_fresh`** | **1,355,121** | 39.35 % |
| **`qs_eval_ttserved`** | **199,811** | **19.82 %** of qs evals |
| **`qs_eval_fresh`** | **808,369** | 80.18 % |
| `main_ttcut` / `qs_ttcut` | 302,799 / 227,918 | |

**2,163,490 fresh evaluations over 5,001,562 nodes = 0.4326 per node.** The
head's own counter reads `eval.calls = 2,156,392`; the 7,098 difference is the
material-draw shortcut and the two `ply >= MAX_PLY` sites, which return before
`forward_into`.

### 1.1 What each fresh evaluation is FOR

The census tags every fresh evaluation with the first consumer that actually
read its value:

| consumer | count | share of that channel's fresh evals |
|---|---:|---:|
| **`main_fresh_rfp`** — reverse futility pruned the node | **977,093** | **72.10 %** |
| `main_fresh_moveloop` — the node reached its move loop | 325,194 | 24.00 % |
| `main_fresh_razor` — razoring dropped it into qsearch | 18,387 | 1.36 % |
| main, remainder (NMP / ProbCut returns) | 34,447 | 2.54 % |
| **`qs_fresh_standpat`** — the stand pat cut | **351,508** | **43.48 %** |
| `qs_fresh_moveloop` | 456,861 | 56.52 % |

The two qsearch rows sum to `qs_eval_fresh` **to the unit**, which is the
census's own consistency check.

> ### **61.41 % of every full NNUE forward pass this engine computes
> (1,328,601 of 2,163,490) is consumed by ONE one-sided threshold comparison
> and then discarded.**
>
> RFP asks `eval − margin >= beta`; the qsearch stand pat asks
> `best_value >= beta`. Neither node has children. Both then *return a blend of
> beta and the eval*, which is why the comparison alone is not enough and the
> exact value is still needed — see §5.

### 1.2 The brief's first direction, measured and closed

> *"How often is a full NNUE forward pass computed that the search does not
> need? Count it."*

Counted. **`main_fresh_then_ttcut = 0`** — the counter never fired. Not one of
the 302,799 main-search TT cutoffs was preceded by a fresh evaluation at that
node. ARCHON orders step 6 (static eval) **before** step 7 (the TT cutoff),
where Stockfish orders them the other way, and that ordering has been a
standing suspicion; it costs **exactly zero evaluations**, because a TT cutoff
requires `tt_hit`, and a `tt_hit` whose eval field is valid serves the eval for
free. **Reordering the two steps would convert 0 % of the traffic.** The
ordering is load-bearing anyway (doc 10's hindsight depth adjustments read
`ss.static_eval` and feed `depth` into the cutoff predicate), so this closes a
route that would have been a functional change for nothing.

**The evaluations the search does not need are not at TT cutoffs. They are the
61.41 % in §1.1, and the search does need their VALUE — it needs it more
cheaply.**

---

## ★★★ 2. WHERE THE LATENCY IS, SPLIT BY STREAM FOR THE FIRST TIME

`tools/perf59/ibs59.sh base 3`, 758,728 sampled ops, 4,143,146 tag-to-retire
cycles (`perf-artifacts-64/ibs-symbols.txt`). IBS tags an op at ISSUE and
records its own rip, so there is no skid; it charges a nonzero weight only to
ops that **miss L1**, so every "latency" column below is an L1-miss service
profile.

| ops % | **lat %** | L1-miss % | cy/op | symbol |
|---:|---:|---:|---:|---|
| 17.60 | **53.61** | 45.26 | 38.0 | `avx512::i8_into_shaped` |
| 13.00 | **14.44** | 14.75 | 13.9 | `SotaHead::forward_into` |
| 3.01 | **6.72** | 6.16 | 27.8 | `HalfkaContext::finny_refresh_into` |
| 15.17 | 4.94 | 5.92 | 4.1 | `Worker::search` |
| 2.10 | 4.11 | 4.03 | 24.4 | `avx512::i8_deltas_into_generic` |
| 1.01 | 3.27 | 1.61 | 40.5 | `avx512::i8_deltas_dual_generic` |
| 1.69 | 2.46 | 1.16 | 18.2 | `Worker::qsearch` |
| 6.32 | 1.70 | 3.26 | 3.4 | `ThreatContext::at_full` |
| 7.54 | 1.18 | 1.76 | 2.0 | `MovePicker::init_quiets` |

(`ops %` is IBS's own sampled-op share under `cnt_ctl=1`, which counts CYCLES
between samples and is therefore time-biased. **It is not an instruction
count** and must not be compared against an `instructions:u` share.)

### 2.1 `foldsrc64.py` — inside the kernel

`perf-artifacts-64/ibs-streams.txt`. Every instruction offset in the fold
kernels classified by the stream it reads: `vpmovsxbw <mem>` is the only
instruction that touches a **weight row**; `vpaddw`/`vpsubw`/`vmovdqu64`
reading memory is the **parent accumulator**; `vmovdqu64 %zmm,(…)` is the
**destination store**.

| kernel | of engine | **WEIGHT ROW** | **SRC acc load** | DST store |
|---|---:|---:|---:|---:|
| `i8_into_shaped` | 53.61 % | **43.10 %** | 10.49 % | 0.00 % |
| `finny_refresh_into` | 6.72 % | **5.34 %** | 0.92 % | 0.00 % |
| `i8_deltas_into_generic` | 4.11 % | **3.80 %** | 0.29 % | 0.00 % |
| `i8_deltas_dual_generic` | 3.27 % | **3.01 %** | 0.26 % | 0.00 % |
| **total** | **67.71 %** | **55.25 %** | **11.96 %** | **0.00 %** |

> **55.25 % of the engine's entire L1-miss latency is weight-row loads.** The
> accumulator STREAM — the parent entry every fold reads and the child every
> fold writes, which is 50.6 % of the fold's BYTES — is **11.96 %**, and
> **96.9 % of it hits L2**. Study 61 §1 predicted exactly this shape ("everything
> else in this engine is essentially L2-resident"); this is the first
> measurement that separates the two streams *inside the same symbol*.

Two consequences, both load-bearing for what is worth building:

* **The write side is invisible.** IBS charges stores zero tag-to-retire, so
  the 19,022 sampled `DST acc store` ops read 0.00 %. That is an instrument
  property, **not** a finding that stores are free — do not quote it as one.
  Any arm aimed at the store side needs a different instrument (study 44's
  store-queue counters).
* **The DRAM tail lives in the weight rows and nowhere else.** By level,
  weight rows are L2 63 % / L3 16 % / **RAM 20 %** in `i8_into_shaped` and
  **RAM 30 % / 48 %** in the two Finny kernels; the accumulator stream is
  **RAM 0.2 %**. The 3.588 DRAM fills/node study 61 measured are all in this
  one stream and they carry **20.4 % of its latency at ~572 cy/op**.

### 2.2 The head, per instruction

`perf-artifacts-64/ibs-streams.txt`, `forward_into` (14.44 % of engine):
**eight `vpdpbusd (%rax,%r11,1)` offsets carry 42.65 % of the head = 6.16 % of
the engine**, ~97 % L2 hits. That is `l1_dot`'s gather — the sparse matmul
whose dense alternative study 61 §6 killed at 31.3 % group density (measured
again here: `nnz` = 216,200,386 groups over 2,156,392 evaluations =
**100.26 of 320**, reproducing study 61 exactly). The `activate_ft` merge
(`vpmullw`/`vpmovzxbw`/`vpsllw` over the four accumulator arrays) is the next
~16 % of the head, also ~100 % L2.

---

## ★★★ 3. THE DEFERRAL MACHINERY IS ALREADY AT 1.13 FOLDS PER EVALUATION

> The brief: *"Accumulator update count, not update cost. How many folds happen
> per node, and how many are avoidable? A fold not performed beats a fold made
> 5 % cheaper."*

Counted, and the answer is **almost none are avoidable**.

`foldcount` calls are per **(ply, perspective)**; dividing by two gives
ply-folds:

| | calls | rows | **ply-folds** | per node | per **evaluation** |
|---|---:|---:|---:|---:|---:|
| `hka_into` | 4,893,314 | 11,150,925 | 2,446,657 | 0.4892 | **1.1346** |
| `thr_into` (materialised) | 4,270,334 | 19,616,896 | 2,284,877 | 0.4568 | **1.0596** |
| `hka_dual` (Finny warm) | 823,842 | 4,712,390 | — | 0.1647 | 0.3820 |
| `thr_refresh` | 98,024 | 407,362 | — | 0.0196 | 0.0455 |
| `hka_inplace` (Finny cold) | 128 | 645 | — | 0.0000 | — |
| **total rows** | | **35,888,218** | | **7.1754** | **16.643** |

`7.1754 rows/node` reproduces studies 47/48/50/51/61 **exactly**.

**Every one of the 5,001,562 made moves pushes a record** (`make.calls =
5,001,562 = nodes`, to the unit) and **48.92 %** of them ever have their
HalfKA fold performed, **45.68 %** their threat fold. Against an eager engine:

| | eager rows (make-time census × 2 perspectives) | folded | **deferral saves** |
|---|---:|---:|---:|
| HalfKA | 22,235,476 | 11,150,925 | **49.85 %** |
| FullThreats | 32,256,496 | 19,616,896 | **39.18 %** |

> ### The bound, stated as a result
>
> An evaluation needs its own ply folded — that is 1.0000 of the 1.1346. **The
> entire avoidable population is the 0.1346 ancestor folds per evaluation, i.e.
> 5.8 % of HalfKA fold-plies and 5.6 % of threat fold-plies**, and those are
> not waste either: an ancestor fold is memoised and reused by every later
> sibling that walks back through it, which is exactly the "backward fill"
> study 12 §6 landed to reach Stockfish parity. **There is no fold-count arm
> here. The brief's hypothesis that a large avoidable fold population exists is
> refuted at 6 %.**

The threat channel's `Fold` machinery is doing the same job: 792,464 of the
3,077,341 folded threat plies (**25.75 %**) resolve to an empty delta for both
perspectives and are **aliased, not copied** (study 38), and a further 109,963
copy one perspective.

---

## ★★ 4. `at_full` IS NOT A FULL REPLAY, AND IT IS 1.70 % OF THE ENGINE

> The brief: *"`at_full` is a full replay where Stockfish maintains
> incrementally… At 8 % of instructions and 10.7 % of L1d misses this is the
> largest structural difference we have."*

**Both halves fail their check.**

**(a) It is not a replay.** `build_deltas` (`sota_threats.rs:2071`) is a
two-endpoint incremental touch generator: for the modal case — a non-capture —
it evaluates exactly two `record_square_touch` calls, at `from` and at `to`,
against the post-move occupancy with both endpoints empty. A capture is
modelled as *mover removal plus a victim→mover identity mutation at the
destination*, again two touches. Only the residual path (castling, en passant)
"replays", and what it replays is the ≤ 4 edits of **one move**, not the
position. Nothing in this channel enumerates the board's threats at fold time
except the deliberate from-scratch rebuild after a king d/e crossing
(`thr_refresh`, **98,024 calls = 0.0196/node**).

**(b) It is small in the currency the brief itself mandates.** Measured:

| | `ThreatContext::at_full` |
|---|---:|
| IBS sampled ops | 6.32 % |
| **L1-missing ops** | **3.26 %** |
| **tag-to-retire latency** | **1.70 %**, at 3.4 cy/op |

Its helper `record_square_touch` adds 0.59 % of latency. The brief's own
closing rule — *"ask what fraction of TRAFFIC a rung converts"* — prices this
whole site at **2.3 %**, against 55.25 % for the weight rows it feeds. The
8 %-of-instructions figure is not contradicted (this study measures no
instruction shares); it is the reason the site *looks* structural and the
reason study 63's lesson applies to it: **these are cheap instructions on a
backend-bound machine.**

The same check on the brief's fourth direction: `search` (all
monomorphisations) is **4.94 % of latency and 5.92 % of L1-missing ops** at
4.1 cy/op — roughly half the 10.8 % of misses the brief quotes, and one
twenty-fifth of its latency weight per op.

---

## ★★★ 5. THE ONLY LEVER LEFT ON ROW COUNT, PRICED

Rows/node = evaluations/node × rows/evaluation = **0.4326 × 16.643**. §3 closed
the second factor. This section prices the first.

**1,328,601 fresh evaluations (61.41 %) are consumed by a threshold test and
nothing else** (§1.1), and those nodes return immediately — they have no
children, so the ply they folded is read by their own head pass and by nobody
else before the next sibling overwrites it. Removing such an evaluation
therefore removes **a whole ply-fold in both channels plus a whole head pass**:

| removed, at the free-predictor ceiling | share |
|---|---:|
| HalfKA ply-folds (1,328,601 of 2,446,657) | 54.30 % |
| threat ply-folds (1,328,601 of 2,284,877) | 58.15 % |
| head passes (1,328,601 of 2,156,392) | 61.61 % |
| × fold latency (weight rows 55.25 % + stream 11.96 %) | 36.5–39.1 % of engine |
| × head latency (14.44 %) | 8.9 % of engine |
| **ceiling** | **≈ 45 % of the engine's L1-miss latency** |

> ### **This is the whole prize, and it is FUNCTIONAL.**
>
> **It cannot be taken bit-exactly.** RFP returns
> `(RFP_RET_BETA_W·beta + RFP_RET_EVAL_W·eval)/RFP_RET_DEN` and the qsearch
> stand pat returns a beta-ward blend of `best_value`; both READ the eval after
> the comparison, so a bound that only decides the comparison cannot produce
> the return value. Razoring is the one consumer that needs the predicate alone
> — and it is **18,387 evaluations, 1.36 %**.
>
> **And the predictor must not touch the big accumulator.** The fold is 82 % of
> what a skipped evaluation saves, and the fold *is* how the value is obtained
> — so a "lazy head" that folds and then skips `forward_into` converts only the
> 8.9 % head share. To convert the 36–39 %, the predictor has to answer before
> the fold, which means **its own accumulator**: the small-net path
> (`smallnet_thresh` / `smallnet_mat_gate` / `smallnet_halfka`) that this engine
> already carries, fully wired and currently inert.
>
> **Stockfish removed its small net (`ab8f901d`), and whether that decision
> transfers is OPEN.** An earlier draft of this section argued it does not,
> "because SF's fold is one channel and ours is two". **That is wrong and is
> retracted** — see §5.1. Our fold is 1.25× SF's, not 2×, so SF's own
> cost/benefit is much closer to ours than that argument claimed, and the
> honest statement is that a cheap predictor has to beat a bar 1.25× higher
> than the one SF judged not worth clearing.

**This is a D-decision and a training task, not a perf arm.** It changes node
counts by construction, so it is arbitrated by games, not by a speed panel —
and pre-registering that here is the point of measuring it.

### 5.1 A RETRACTION, and a correction to a doc that landed mid-study

`main` moved to `802cbb98` while this study was open. Two of its commits
(`dbd43bb3`, `4a6d5f71`, both editing `docs/research/d48y-wdl-fix-plan.md`)
compare our architecture against Stockfish's and the owner rules on it. Read
against the source rather than against either document
(`eca43a97:src/nnue/nnue_architecture.h`, lines 39–49 and 60–70):

```
using ThreatFeatureSet = Features::FullThreats;      // :39
using PSQFeatureSet    = Features::HalfKAv2_hm;      // :40
constexpr IndexType TransformedFeatureDimensionsBig = 1024;   // :43
constexpr int       L2Big = 15;                               // :44
constexpr int       L3Big = 32;                               // :45
...
Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS + 1> fc_0;
Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>                            fc_1;
```

**RETRACTED (mine).** I wrote that "SF's fold is ONE channel, ours is two".
**Stockfish runs `Features::FullThreats` as well**, and `dbd43bb3` is right
that the input spaces are identical. The retraction weakens §5's closing
argument and I have rewritten it above rather than quietly dropping it.

**CORRECTION (`dbd43bb3`).** That commit prices SF's `fc_0` at **262,144
against our 163,840, "SF 1.600×"**, and builds a two-cost-regimes story on it
("SF spends 1.6× more on the free layer to spend 0.8× on the expensive one").
**SF's `fc_0` output width is `FC_0_OUTPUTS + 1` = 15 + 1 = 16, not 32** — the
32 is `L3Big`, i.e. `fc_1`'s output. So SF's `fc_0` is `1024 × 16 × 8 buckets =
131,072` against our `1280 × 16 × 8 = 163,840`: **SF is 0.800×, the same 0.800×
as its feature transformer, not 1.600×.** Confirmed on our side from
`sota_head.rs`: `L1 = 1280`, `L2 = 16`, `L3 = 32`, `BUCKETS = 8`, and
`l1_weights` measures 163,840 bytes in the region census.

> **The corrected comparison is simpler and stronger than the one it replaces.**
> Every layer of Stockfish's big net is our layer at **exactly 1024/1280 =
> 0.800×**: same two feature sets, same 16-wide `fc_0`, same 32-wide `fc_1`,
> same 8 buckets. There is no trade between a cheap layer and an expensive one
> — there is one width, applied everywhere. **The measured 1.263× speed ratio
> against a 1.250× width ratio is the whole story**, and it is the same
> conclusion `dbd43bb3` reached, now resting on one number instead of two
> offsetting ones. (Our `skip_weights` lane is the one real shape difference:
> SF gets it free as `fc_0`'s 16th row, we carry it as a separate 1280-byte
> dot — which `forward_into`'s own comment already says.)
>
> It also puts a caveat on `4a6d5f71`'s "L2 remains a separate open dial …
> doubling 16→32 costs almost nothing in traffic". Doubling `L2` doubles
> `l1_weights` from 20,480 to 40,960 bytes per bucket, and §2.2 measures that
> table's gather at **6.16 % of engine latency at ~97 % L2 hits** on a 91.44 %
> L1-miss stream (study 51). It is cheap, not free, and it would be the second
> time this table doubled.

**The owner ruling (`4a6d5f71`) is what this study was already assuming**: L1 =
1280 stands and may grow, and traffic-reduction work is the lever a wider
accumulator does not blunt. Note what that does to §6.1: **narrowing the STORED
row is the only byte-cut that is compatible with keeping — or growing — L1**,
because it cuts bytes per lane rather than lanes per row. It is now the one
open route in the 55.25 %.

---

## ★★ 6. FOUR ROUTES PRICED AND CLOSED, EACH WITH ITS ARITHMETIC

Every one of these was killed by a counter or by arithmetic over measured
numbers, **before** a line of arm code was written (rule 1).

### 6.1 Narrow weight rows — the static table lies by 1.6×

Study 61 left exactly one variable open: *"the only free variable left in
`bytes ÷ 1280` is the 1280."* `FOLDROWBITS` answers it over the population that
matters. A row whose peak magnitude is `m` needs `ceil(log2(m+1)) + 1` signed
bits, so a ≤ 6-bit row can be stored in 6 bits **losslessly**.

| | ≤ 6 bits | 7 bits | 8 bits |
|---|---:|---:|---:|
| threat table, by ROW (static) | 52,472 / 60,720 = **86.4 %** | 12.8 % | 0.8 % |
| threat, by **FOLD** (19,616,896 rows) | 10,409,593 = **53.06 %** | 38.71 % | 8.23 % |
| HalfKA, by FOLD (11,150,925 rows) | 0 | 2.70 % | **97.30 %** |

> **The folded population is 1.63× more biased toward wide rows than the table
> is** — 86.4 % of *rows* fit in 6 bits but only 53.1 % of *folds* do — which is
> CLAUDE.md rule 7 (per-class and per-position, together) landing on a route
> that would otherwise have been costed at nearly double its value.

Priced: 320 B saved on 10,409,593 rows = **3.331 GB of the 45.94 GB** the frame
folds = **7.25 % of all weight-row bytes**. HalfKA contributes nothing. The
cost is a two-plane layout plus a 60,720-entry permutation table (243 KB) read
3.92×/node, and a 6→16-bit AVX-512 unpack (`vpermb` + `vpmultishiftqb` + mask +
sign-fix, ≈ 4 extra shuffle-port ops per 64 lanes, ≈ +80 per row).
**Not built** — it is the largest bit-exact route left and it needs a study of
its own; see §9 item 1.

### 6.2 All-zero rows — 30 in 19.6 million

`FOLDZEROROW` on the folded population: **hka 0 of 11,150,925; thr 30 of
19,616,896.** The static table has 9,976 all-zero threat rows (16.4 %) and
1,024 HalfKA (4.5 %) — **the zeros sit entirely in features that never occur.**
This re-verifies the in-tree banner on `foldcount::ZERO_ROW_*` from an
independent run and closes the route for good.

### 6.3 Delta cancellation — 1.99 %

`thr_cancel`: 194,875 cancel pairs over 3,077,341 folded plies, i.e. **389,750
of 19,616,896 threat rows = 1.99 %**, and `dup_add = dup_sub = 0`. Detecting
them costs a quadratic scan of the index lists inside the hot fold. Dead.

### 6.4 From-scratch rebuild instead of a long fold — dead by arithmetic

The threat delta reaches 27 rows, and a from-scratch rebuild is a fixed
**10.702 active rows per perspective** (`active_thr` = 46,155,264 rows over
4,312,784 = 2 × 2,156,392 calls). Rebuilding also skips the 2560-byte parent
read (2 row-equivalents). Crossover at `k > 8.7`:

| threshold | folds | rows today | rows rebuilt | saved (row-equiv) | **saved** | rebuild cost |
|---|---:|---:|---:|---:|---:|---:|
| `k >= 9` | 615,920 | 6,985,617 | 6,590,344 | 1,627,113 | **2.50 % of engine** | **3.34 %** |
| `k >= 15` | 66,602 | 1,095,639 | 712,641 | 516,202 | 0.79 % | 0.36 % |

The rebuild cost is measured, not modelled: `refresh_threat_into_at` is
**0.531 % of engine latency at 98,024 calls**, so 615,920 calls cost 3.34 %.
**Net −0.84 % at `k >= 9`; +0.43 % at `k >= 15`, which is under the 1.57 %
cycles noise floor.** Dead at both.

### 6.5 Co-locating the PSQ1 row with its accumulator row — dead by arithmetic

Every folded row also pulls one random 32-byte PSQ1 line out of a separate
1.94 MB table (7.175 such reads/node). Packing the 32 PSQ1 bytes into the
accumulator row would make that read free — but the row goes 1280 B (20 aligned
lines) → 1312 B, which must pad to **1344 B (21 lines)** to preserve study 61's
assumption A1. **+5.00 % of table bytes against a 1-in-21 = 4.76 % saving. Net
−0.24 %.** Dead before a build.

---

## ★ 7. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **No wall-clock number, and therefore no Elo.** Deliberate: study 63 found
   four defects in `wallclock_panel.py` and showed its three valid runs are
   mutually inconsistent. Nothing landed here changes behaviour, so there was
   nothing for it to arbitrate — but it also means **the §5 ceiling is a
   latency-share arithmetic, not a measured speedup**, and the 45 % should be
   read as an upper bound with the assumptions in that section attached.
2. **The §5 ceiling assumes fold latency is uniform per ply-fold.** It is not
   exactly: a leaf's fold has the same shape as any other, but its weight rows
   may have a different reuse distance from an interior node's. I did not
   measure the leaf/interior split of §2.1's latency, and the honest bracket is
   therefore wider than the 36.5–39.1 % quoted.
3. **IBS `ops %` is time-biased** (`cnt_ctl=1` counts cycles between samples),
   so it must not be compared against `instructions:u` shares. Where the brief
   quotes instruction shares (26.95 % for `search`, 8 % for `at_full`) I have
   compared only the **miss** and **latency** columns, which are not biased
   this way. The instruction figures are neither confirmed nor contradicted.
4. **`DST acc store` reads 0.00 % because IBS does not weight stores**, not
   because the fold's 2560-byte-per-call write side is free. §2.1 says so; a
   successor must not read that row as a closure.
5. **`thr_refresh.rows / calls = 4.16` against `active_thr = 10.702`.** Those
   two ought to be the same quantity (rows a from-scratch rebuild folds) and
   are not. I used the `active_thr` figure in §6.4 because it is the enumeration
   census and the more conservative of the two for that arm's *benefit* side;
   the discrepancy is unexplained and someone should resolve it before quoting
   either as "the from-scratch row count".
6. **No games.** Nothing landed changes behaviour, so there is nothing to play.
   The one arm worth playing (§5) is not built.
7. **I published a wrong claim about Stockfish's architecture and caught it
   only because `main` moved.** §5's first draft said SF runs one feature
   channel to our two; it runs both (§5.1). I inferred it from CLAUDE.md's
   perf-study-7 banner instead of opening
   `eca43a97:src/nnue/nnue_architecture.h`, which is precisely the failure
   CLAUDE.md's Stockfish section warns about. The correction is in §5.1 and the
   argument that rested on it is rewritten, not deleted.
8. **The 6-bit unpack cost (§6.1) is an instruction estimate, not a
   measurement.** Whether ≈ 80 extra shuffle-port ops per row are absorbed by a
   backend-bound machine (study 63's finding) or serialise on Zen 4's single
   shuffle port is exactly the question that arm has to answer, and I did not
   answer it.

---

## ★ 8. GATES

| gate | result |
|---|---|
| **shipped `.text` unchanged** — `base2` rebuilt from the final tree vs `base` from `4d41d407` | **md5 `75c9955e` = `75c9955e`, size `760,085` = `760,085`**; whole-frame `instructions:u` 27,346,635,514 vs 27,346,635,453 = **2.2 parts in 10⁹** |
| node identity, study frame, every binary (5 built) | **5,001,562** — exact |
| wl50 eval hash, every binary incl. the instrument | **`2e1ede95…79c4`** — exact |
| `foldcount` rows/node vs studies 47/48/50/51/61 | **7.17540** — exact |
| `nnz` group density vs study 61 §6 | **100.26 of 320 (31.3 %)** — exact |
| `thr_pair` shape census vs studies 61 §4.1 / 63 §5 | 4,270,334 threat folds, `hka_into` cover **100.00 %** — exact |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | **exit 0 — `All checks passed.`, `Bench: 587683`, eval hash `2e1ede95…79c4`, d13 nodes `1739588`; all three pins unmoved** |
| `cargo test --release --test perft_suite -- --ignored perft_full_suite` | **ok** (1 passed) |
| `cargo clippy --all-targets` (default) and `--features sota,arch-d48x,foldcount` | **clean, both** |
| `cargo fmt` | clean |
| rebase re-gate (study 56's `bcmisc55` rule) | opened on `4d41d407`; `main` moved to **`802cbb98`** during the study. `git diff --stat 4d41d407 802cbb98` is **one docs file and no `src/`, no `Cargo.toml`**, so no measurement expired. Rebased; the gate row above was run on the rebased tree, and `base3` (rebuilt from the final tree) still reproduces `.text` md5 `75c9955e`. Two of those commits make architecture claims this study checked against the source — §5.1 retracts one of mine and corrects one of theirs. |

D24 perft ritual not strictly required (nothing under `movegen`, `board`
mutation, `zobrist` or `attacks` is touched) — run anyway.

---

## ★ 9. WHAT THE NEXT STUDY SHOULD KNOW

1. **The narrow-row route is the largest BIT-EXACT thing left, and it is worth
   7.25 % of weight-row bytes — not 25 %.** §6.1 has the folded-population
   histogram, which is the number that decides it; do not re-derive it from
   `FOLDROWSTATS`, which overstates by 1.63×. Study 61's capacity model says a
   byte cut is convex (a 20 % byte cut was a 30 % cut in L2-missing bytes), so
   the latency payoff is plausibly ~1.4× the byte figure — but the unpack cost
   is unmeasured (§7 item 7) and that is the first thing to panel.
   **`4a6d5f71` makes this the ONLY open route in the 55.25 %**: with L1 = 1280
   ruled permanent and possibly growing, bytes-per-lane is the one term in
   `rows × lanes × bytes` still free to move.
2. **The 45 % prize is a NET decision, not a kernel one** (§5). If anyone
   proposes work on the fold kernels again, the ceiling on all of it is
   `55.25 % × (whatever fraction of ROW BYTES the arm removes)`, and every
   locality, ordering, dispatch and prefetch route into that 55.25 % is closed
   (studies 44/51/60/61/63 plus §6.2–6.5 here).
3. **Charge latency to a STREAM, not to a symbol.** `tools/perf64/foldsrc64.py`
   turns "the fold is 53.6 %" into "weight rows 43.1 %, accumulator stream
   10.5 %, stores unmeasurable", and those three have completely different
   levers. It works on any symbol; run it before proposing anything inside one.
4. **Charge evaluations to their CONSUMER.** `foldcount::SEARCH` is 17 hooks
   and one array; it is what turned "43 % of nodes evaluate" into "61.4 % of
   evaluations answer a threshold". Extend it rather than rebuilding it.
5. **Three of the brief's four named directions priced out at ≤ 6 %** (§1.2,
   §3, §4). That is not an argument that the briefs are wrong — it is the
   consequence of ranking by instruction share, which studies 61–63 have now
   shown three separate times does not predict where the machine waits.
6. **Still open, inherited and untouched:** study 43 §9 item 2's subtractive
   Finny-line arm (an instrument, not a speedup); study 57 §6's FNV-1a-64
   payload checksum (~45 % of startup, needs a format ruling); study 48 §9
   item 3's AVX2 `activate` hoist; study 63 §7 item 1's re-run of the +3.14 Elo
   receipt on the repaired wall-clock tool.

---

## ★ 10. REPRODUCE

```bash
ln -s /path/to/chessengine/data data          # the worktree needs the net

bash tools/perf61/build61.sh base                     # the shipped control
bash tools/perf61/build61.sh fcount2 foldcount        # §1 + §3 + §6.2-6.4
bash tools/perf61/build61.sh fcrow   foldcount,rowprobe  # §6.1 + §6.2

# §1 — the search-side work census (and everything in §3)
ARCHON_FOLDCOUNT=/var/tmp/s64-fc2.txt taskset -c 2 ./bin61/fcount2 \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep 'search\.' /var/tmp/s64-fc2.txt

# §6.1 / §6.2 — the FOLDED-population bit width and zero-row census
ARCHON_FOLDCOUNT=/var/tmp/s64-fcrow.txt taskset -c 2 ./bin61/fcrow \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep -E 'FOLDROWBITS|FOLDZEROROW' /var/tmp/s64-fcrow.txt

# §2 — the latency profile, then the stream split inside each kernel
mkdir -p bin59 && cp bin61/base bin59/base
P59_CPU=2 bash tools/perf59/ibs59.sh base 3 /var/tmp/s64-ibs.txt
python3 tools/perf61/ibstop61.py /var/tmp/s64-ibs.txt 22
python3 tools/perf64/foldsrc64.py /var/tmp/s64-ibs.txt bin61/base i8_into_shaped
python3 tools/perf64/foldsrc64.py /var/tmp/s64-ibs.txt bin61/base finny_refresh_into

# §8 — the instrument must be invisible to the shipped build
bash tools/perf61/build61.sh base2                    # expect md5 75c9955e
P61_CPU=2 bash tools/perf61/ident61.sh base base2 fcount2
CHECK_NNUE_GATE=1 bash tools/check.sh
cargo test --release --test perft_suite -- --ignored perft_full_suite
```

**Do not run a panel while anything else builds**, and do not park perf text on
`/tmp` — it is tmpfs.
