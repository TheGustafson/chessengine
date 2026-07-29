# NNUE architecture study — is D48X optimal, and if not, what is?

**Commissioned 2026-07-28. Research deliverable only: no `src/` change, no training run.**
Artifacts and reproduction scripts: `docs/research/nnue-arch-artifacts/`.

Every number below is tagged **[M] measured** (someone ran it and recorded it),
**[D] derived** (arithmetic on measured inputs, with the inputs cited), or
**[E] estimated** (a judgement, with the basis stated). The failure mode this repo
names explicitly — an estimate hardening into a fact across three documents — is the
reason for the tagging, and §11 lists the three places where I found it had already
happened.

---

## 0. HEADLINE

**(1) The architecture is close to a local optimum, and the three biggest levers
are not width, not the neck, and not the feature set.** They are (a) an *unconsumed
output the net already produces*, (b) a *tail shape Stockfish adopted after we copied
its predecessor*, and (c) *output-head routing*, which is the only capacity axis in
this architecture that is free at inference time.

**(2) The exchange rate that prices every architecture decision in this project —
"~120 Elo per doubling of speed" — has a 95% confidence interval that includes
zero.** [D] It comes from one 100-game match (exp-0154, +38.4 ± 48.0) differenced
against one 200-game fixed-node match (exp-0149, −34.9 ± 26.0), a 73.3 Elo swing
with a ±54.6 Elo 1σ bar, divided by 0.54–0.59 doublings ⇒ **120–136 Elo/doubling,
±~95 (1σ)**. Every conclusion in the D48W→D48X width decision, the threat-channel
debate and this study's own cost tables is a function of that constant, and at the
low end of its band (70/doubling) *the width decision reverses sign*. **Re-measuring
it is the cheapest high-leverage experiment on the board and it needs no GPU.**

**(3) A cost model for the architecture knobs now exists and validates to ≤3% against
the only independent speed ladder we have.** [D] `docs/research/nnue-arch-artifacts/cost_model.py`,
calibrated on perf study 8's instruction decomposition, reproduces all four points of
exp-0156's idle-box PGO NPS smoke (768/32, 1280/16, 1536/16, 1280/128) to +0.6%,
0.0%, +1.9%, +2.9%. It is the instrument the rest of this study prices options with.

**(4) The owner's "512 → 8 → 8 and be super fast" is 94% a width proposal wearing a
head-shape costume, and the head-shape part buys ~4 Elo.** [D] Decomposed with the
model: 1280→512 is +54.8 Elo of speed at 120/doubling; 16/32 → 8/8 on top of that is
**+4.9**. The entire post-`l1_dot` tail is 4.7% of engine instructions, so it cannot
make anything "super fast". And the width half runs straight into the standing
width-neutrality finding: −58 fn Elo at +44/doubling against +54.8 of speed ⇒ a wash.
**Rejected, with the arithmetic in §6.**

**(5) Stockfish at our own constants reference (`eca43a97`, 2026-07-06) has NO small
net.** [M, verified by me directly: `git grep -il "smallnet\|NetworkSmall" eca43a97 -- src/`
returns *nothing*; the same grep at `sf_18` returns three files.] The 962/277 dual-net
scheme is a `sf_18`-tag feature that master deleted. The owner's "dual big/small nets
as SF has" is answered: **SF had it and removed it.** Perf study 7 had already priced
its value for SF at 3.65–6.50% of instructions [M].

**(6) The same commit tells us something more useful: SF's *tail* got wider and
deeper while ours did not.** [M, verified: `eca43a97:src/nnue/nnue_architecture.h:43-45,56-67`]
`L2` went 15 → **32**, a second dual activation was added, and `fc_2` now reads a
**128-wide concatenation of both hidden layers' activations** — an explicit skip from
layer 0 straight to the output. We copied the `sf_18`-era shape (L2=15/16, single
CReLU tail). SF spends **1.57× our per-eval head MACs at 0.8× our accumulator width** [D].

**(7) Our neck is not the bottleneck — the *tail's ability to use it* is.** [M, new
measurement, with its null] The participation ratio of the L2 weight matrix
(neck → L3) is **9.95 at L2=16, 11.06 at L2=32, and 6.22 at L2=128** across three
trained ARCHON nets. Widening the neck 8× does not widen what reaches L3; it *narrows*
it. The null panel (i.i.d. resampled matrices of identical shape) gives 11.83 / 18.35 /
26.14, so the observed collapse at 128 is 4× below its null and unambiguous. **This
retires the standing suspicion, recorded in exp-0122's notes, that "L2=16 is the
bottleneck for width-saturation."**

**(8) The field says our tail is the one place we are genuinely behind, and someone
else has already measured the fix.** A survey of thirteen top-20 engines [S, §5.4] finds
**eleven carry a dual concatenated activation (CReLU ‖ SqrCReLU); ARCHON, Reckless and
Integral are the three that do not.** Stormphrax measured it standalone at **+6.95 fixed
nodes / +6.20 STC / +12.05 LTC** (PR #239). That turns §0(6)/(7) from a reasoned
inference into the best-evidenced item in this study.

**(9) Two of the survey's findings say our *test conditions* are wrong, not our
architecture.** Pawnocchio PR #68 measured a width increase at five conditions —
**+18.10 ± 5.64 fixed nodes, −11.02 at 10+0.1, −7.16 at 40+0.4, +5.02 at 60+0.6×8t** —
and every threat-feature result in the field has the same shape. **Our whole arch-testing
regime (fn50k and 10+0.1, Threads 1–4) sits on the side of that crossover where a better
evaluator reads as a loss.** Combined with `searchopt/LEDGER.md`'s ±40–50 Elo half-width
at n=406, several standing architecture verdicts here are underpowered *and* measured in
the wrong regime.

**(10) I killed my own most attractive idea with a measurement.** A Vine-style
"threat as two bits on the piece-square feature" would have deleted ~18% of our
instructions and unified our two threat systems — but an ANOVA over the trained threat
table (with its null) says **no coarse grouping of the relation explains more than ~5%
of its row variance above null**; the learned content is in the specific
(attacker, from, victim, to) relation. Recorded in §8.5 rather than recommended.

**(11) The honest overall verdict: D48X's *shape* is defensible, its known deficiencies
are implementation rather than architecture, and its one real architectural gap is the
tail.** Perf study 8's decomposition says the 1280 width is fully paid for (1.080×
justified premium, both derivations agreeing to 0.3%) and that the residual 1.40× is a
replay stage, a duplicated threat system, an unprofiled make/unmake and accumulator
memcpys. **A redesign is not what stands between us and SF18.** What this study
recommends are small, cheap, individually falsifiable changes — not a new architecture.

---

## 1. THE MEASURED BASELINE — what D48X actually is

### 1.1 Layer by layer

Read from the net file's own header and payload
(`docs/research/nnue-arch-artifacts/net_anatomy.py`, which parses CENN-v2 directly)
and from `src/eval/nnue/sota_head.rs` / `sota_features.rs`.

| stage | shape | dtype | source |
|---|---|---|---|
| HalfKAv2_hm features | 22,528 dims, 32 king buckets, file-mirror to e–h | — | `sota_features.rs:56,64-75,110` |
| FullThreats features | 60,720 dims, mirror only (no king bucket) | — | `sota_features.rs:135` |
| HalfKA transformer | 22,528 × 1280 | i16 | header `acc=1280` |
| threat transformer | 60,720 × 1280 | i8 (`THREAT_QUANT=64`) | `sota_head.rs:227` |
| FT bias | 1280 | i16 | |
| pairwise-mul activation | (2 × 1280 i16) → 1280 u8, `>>9`, STM half first | u8 ∈ [0,127] | `sota_head.rs:196-203` |
| **L1 (neck)** | 1280 → **16**, NNZ-sparse `vpdpbusd` | i8 weights, `L1_QUANT=64` | `sota_head.rs:97` |
| L2 | 16 → 32, CReLU | f32 | `sota_head.rs:104` |
| L3 | 32 → 1 | f32 | `sota_head.rs:104` |
| output-skip lane (D47V) | 1280 → 1, per bucket | i8 since perf study 8 §4.6 | `sota_head.rs:430-433` |
| PSQ1 skip (T6c) | 83,248 → 8, unclamped, added in cp | i32 | `sota_head.rs:437` |
| output buckets | 8, non-uniform `OUTPUT_BUCKETS_LAYOUT[popcount]` | — | `sota_head.rs:130-140` |
| `network_scale` | 400 | | header |

Optional, present in the loader and **absent from the shipped net**: a heteroscedastic
log-variance head (`l3_var_weights`, `sota_head.rs:171`) and a 3-logit WDL head
(`l3_wdl_weights`, `sota_head.rs:180`), both read from sidecars, both `None` today.

### 1.2 Where the parameters are

[M/D, `net_anatomy.py` on `engines/canonical/net.cenn`, payload length reconciled
exactly against the header's `payload_len`]

| block | params | bytes | share |
|---|---:|---:|---:|
| HalfKA transformer | 28,835,840 | 57,671,680 | 41.7% |
| **threat transformer** | **77,721,600** | **77,721,600** | **56.2%** |
| FT bias | 1,280 | 2,560 | — |
| PSQ1 trailer | 665,984 | 2,663,936 | 1.9% |
| output-skip | 10,240 | 40,960 | 0.03% |
| **head (all 8 buckets)** | **168,584** | **182,792** | **0.13%** |
| **total** | **107.40 M** | **138,283,640** | |

**The head is 0.157% of the net.** [D] Per eval exactly one bucket is touched:
**22,353 parameters** (20,480 L1 + 16 + 512 L2 + 32 + 32 L3 + 1 + 1,280 skip).

This is the same shape Stockfish has — at `sf_18` the big net's dense stack is 0.13%
of its 105.8 M parameters — and it is the fact that governs everything downstream:
**an NNUE is a giant sparse embedding table with a rounding error bolted on the end.**

### 1.3 Where the instructions are

[M] Perf study 8 §2 and §3.2, matched 15.0 M-node wl50 panel, `instructions:u`
spread 0.0000% over six interleaved rounds, D48X PGO on the AVX-512 VBMI2 path.
Baseline **7,116.6 instructions/node** against SF18's **4,330.6** (1.6433×).
`evals/node = 0.366` [D, from `activate`'s 351.6 i/n over a 960-instruction kernel].

| block | % of engine | i/n | scales with |
|---|---:|---:|---|
| `activate` (pairwise FT merge) | 4.94 | 351.6 | L1 |
| `find_nnz` | 0.75 | 53.4 | L1 |
| **`l1_dot`** | **6.17** | **439.1** | L1 × (7 + L2/16)/8 |
| `l1_dequant` | 0.04 | 2.8 | L2 |
| `l2_tail` + `l3` | 2.16 | 153.7 | L2·L3 + L3 |
| `skip_dot` | 2.50 | 177.9 | L1 |
| **head forward, total** | **16.55** | **1,177.8** | |
| `at_full` — threat fold | 10.64 | 757.2 | L1 |
| `at_full` — replay plumbing | 5.73 | 407.8 | events |
| `current_full` (HalfKA) | 6.67 | 474.7 | L1 |
| `i16_deltas` | 3.45 | 245.5 | L1 |
| `record_square_touch` + `record_targets` | 3.55 | 252.6 | events |
| `finny_refresh_into` | 1.86 | 132.4 | L1 |
| `refresh_threat` | 0.91 | 64.8 | L1 |
| accumulator `memmove` | 2.00 | 142.3 | L1 |
| **accumulator maintenance, total** | **34.81** | **2,477** | |
| eval layer total | 50.5 | 3,596.7 | |
| search shell | 33.6 | 2,389.0 | |
| make / unmake | 11.0 | 782.8 | |

**Two-thirds of our eval cost is accumulator maintenance and one-third is the head
forward.** [D] That single ratio is why "shrink the head" cannot be a speed strategy
here, and it is the arithmetic that answers the owner's `512 → 8 → 8` question (§6).

### 1.4 Where the Elo currently comes from — and where it does not

[M] From `searchopt/LEDGER.md` (2026-07-28, full 404-game D48X corpus, all
detector-independent):

- **Our static eval is at par with SF18's or better**: MAE **97.1 vs 97.2**, corr
  0.899 vs 0.905, our p90 error **lower** (149.5 vs 176), n=630 across all 404 corpus
  games. This is an AXIOM for this study, not a hypothesis.
- **The equal-nodes deficit is ~68 Elo and saturates by 30k nodes**; at 110k our depth
  equals SF18's (16.0 vs 16.0, n=8,219 moves).
- **We do not lose to blunders**: 25 in 5,379 still-playable moves (0.46%), zero moves
  in the whole corpus at wp_loss ≥ 0.20.
- **We lose because we cannot convert**: on the same 201 colour-paired openings we
  convert a won book **12%** and SF converts **84%**; squandered wins gain a median
  **+0.05** pawns over the book and **none ever exceeds +2.00**.
- **The bleed is distributed and middlegame-localised**: median worst single move per
  squandered game is 0.25 pawns; drift peaks at plies 30–39 and is over by ply ~100.
- **Our root churns 1.561× SF18's in balanced positions** (|score| < 0.25) and only
  1.107× once |score| > 1.0.
- **The in-game speed gap is ~1.4×**, not 1.92× (four independent in-game
  measurements, n≈27–36k moves each, plus perf study 8's 1.512× instructions/node).

**The shape of the defect is therefore: correct on average, under-resolved in the band
where games are decided.** Hold that; it is what picks the winners in §7.

### 1.5 The promotion's own falsifiable prediction, and what this study owes it

`engines/canonical/MANIFEST.md` records D48X's promotion at **−16.6 [−43.5, +10.2] Elo
over 336 games** (updated to −9.1 [−34.4, +16.1] at 382 games in the promotion commit)
— a deliberate promotion on the ruling that D48X is the stronger *eval* and loses only
because the search cannot exploit it. **The prediction it makes is that D48X's margin
over D47V should widen as search quality improves**, and the manifest asks for
`engines/runs/d48x-vs-v` to be re-run after any landed search gain.

**Three items in this study are exactly such gains** — R1 (complexity → optimism, a
pure search-side change), R4 (incremental threats, node-identical speed), and perf
study 8's six landed changes (−8.02% instructions). **Each should re-run that match and
record the delta in the manifest.** If the margin does not widen after all of them, the
manifest's own standing ruling applies: *do not revert to D47V; retrain at a smaller
accumulator*. This study's §4.1 disagrees with the "smaller accumulator" half of that
ruling on the arithmetic — the width is fully paid for and 512 is a wash — and would
instead point at R2/R3/R5, which buy eval quality at ~zero per-node cost. That is a
disagreement to settle with R0's constant, not with an argument.

---

## 2. THE EXCHANGE RATE — the number everything else is denominated in

Every architecture trade in this project is "eval quality vs per-node cost", converted
by Elo-per-doubling-of-speed. The project's working constant is **~120**.

**Provenance** [M]: exp-0154, V (768/32) vs S-post (1280/128), 10+0.1, Threads 4, both
PGO, **100 games**: **+38.4 ± 48.0**. exp-0149, the same pair at fn50k, **200 games**:
**−34.9 ± 26.0**. The 73.3 Elo swing is attributed entirely to the speed ratio
(1.451× by exp-0156's NPS ladder, 1.51× as quoted in exp-0154's note).

**Derivation and its bar** [D]: 73.3 / log₂(1.451) = **136 Elo/doubling**; 73.3 /
log₂(1.51) = **120**. Propagating both matches' errors: σ = √(48.0² + 26.0²) = 54.6
Elo on the swing ⇒ **σ ≈ 92–102 Elo/doubling**. The 95% interval spans roughly
**0 to 320**.

**Why this matters more than any single architecture choice.** The width decision is
`+44 fn Elo/doubling` (eval) against `−45 to −65 Elo/doubling` (speed, §3):

| assumed Elo/doubling | speed cost of one width doubling | eval gain | net |
|---|---:|---:|---:|
| 120 (project constant) | −49.5 | +44 | **−5.5 (neutral)** |
| 70 | −28.9 | +44 | **+15 (widen)** |
| 170 | −70.2 | +44 | **−26 (narrow)** |

**The same evidence supports "widen", "hold" and "narrow" depending on a constant
whose CI includes zero.** The D48W→D48X decision, the threat-channel EV, and every
row of §3's table are all sitting on it.

**Recommendation R0 (§7): measure it directly.** Same binary, same net, two time
controls a factor of two apart (10+0.1 vs 5+0.05), n ≥ 2,000, or the cleaner
`nodes=N` vs `nodes=2N` form which removes time-management confounds entirely. No
GPU, no engineering, ~1 box-day. Until then **every Elo figure in this document that
derives from a speed change is quoted in both frames (120 and 70)**.

---

## 3. THE COST MODEL — derived, then validated

`docs/research/nnue-arch-artifacts/cost_model.py`.

**Construction** [D]. Instructions/node is decomposed into: a width-independent
remainder (shell, movegen, movepick, make/unmake), an accumulator-maintenance block
linear in L1, a threat-event block independent of L1, and the six head phases with
their individual scalings from §1.3. The one non-obvious scaling is `l1_dot`: perf
study 8 §4.1 disassembles its loop body as **eight static instructions of which
exactly one (`vpdpbusd`) scales with L2 in 16-lane steps**, so `l1_dot ∝ L1 · (7 +
L2/16)/8` — not `∝ L1 · L2`. Assuming the naive linear form over-predicts the
1280/128 point by 24%.

**Validation** [M vs D] against exp-0156's idle-box PGO NPS ladder (untrained warm
packs — speed is an arch property, not a weights property — 4 s movetime, 2 positions,
all on the AVX-512 VBMI2 path; W reproduced to 0.01%):

| arch | measured NPS | measured ratio vs X | model | error |
|---|---:|---:|---:|---:|
| V 768/32/32 | 1,051k | 1.177 | 1.184 | **+0.6%** |
| X 1280/16/32 | 893k | 1.000 | 1.000 | 0.0% |
| W 1536/16/32 | 804k | 0.900 | 0.917 | **+1.9%** |
| S 1280/128/32 | 724k | 0.811 | 0.834 | **+2.9%** |

Three of the four points vary a knob the model was **not** calibrated on. Mean absolute
error 1.4%.

**Caveats, stated because the model is used for ranking below.** (i) It is calibrated
on the pre-perf-study-8 `main` (7,116.6 i/n); study 8's branch is at 6,546 i/n and
would rescale every absolute number by ~0.92 without changing any ratio materially.
(ii) It is an *instruction* model. Perf study 8 §7.11 is the standing counter-example —
storing threat rows at i16 cut instructions 2.10% and **lost** on cache — so any option
that makes a table bigger needs `cache8.sh`, not this model. (iii) exp-0156's ladder is
a cold 2-position movetime probe, the regime `searchopt/LEDGER.md` warns is not the
game regime; it is used here only for *ratios between our own arches*, which is the one
thing it is good for.

**The option table** (`cost_model.py table()`):

| arch | i/n | vs D48X | speed | Elo@120 | Elo@70 |
|---|---:|---:|---:|---:|---:|
| D48X 1280/16/32 (canonical) | 7,117 | 1.000 | 1.000 | 0.0 | 0.0 |
| D47V 768/32/32 | 6,010 | 0.845 | 1.184 | +29.2 | +17.1 |
| **owner's 512/8/8** | **5,041** | 0.708 | 1.412 | **+59.7** | **+34.8** |
| 512/8/8, threats OFF | 4,212 | 0.592 | 1.690 | +90.8 | +53.0 |
| 512/16/32 | 5,187 | 0.729 | 1.372 | +54.8 | +31.9 |
| neck 16 → 8 | 7,015 | 0.986 | 1.014 | +2.5 | +1.4 |
| neck 16 → 32 | 7,319 | 1.028 | 0.972 | −4.9 | −2.8 |
| neck 16 → 128 | 8,533 | 1.199 | 0.834 | −31.4 | −18.3 |
| width → 2048 | 9,047 | 1.271 | 0.787 | −41.5 | −24.2 |
| width → 3072 | 11,620 | 1.633 | 0.612 | −84.9 | −49.5 |
| threats OFF @1280 | 5,422 | 0.762 | 1.312 | +47.1 | +27.5 |
| threats OFF @2048 | 6,488 | 0.912 | 1.097 | +16.0 | +9.3 |
| threats OFF @2560 | 7,198 | 1.011 | 0.989 | −2.0 | −1.1 |
| threats OFF @3072 | 7,908 | 1.111 | 0.900 | −18.2 | −10.6 |

The threats-OFF row at 1280 predicts 1.312× against perf study 7's measured **1.317×**
(wl50) / 1.389× (middlegame) — that block was calibrated to the wl50 figure, so it is a
consistency check rather than an independent validation.

---

## 4. THE DESIGN SPACE, OPTION BY OPTION

### 4.1 Accumulator width — 512 / 1280 / 3072

**Evidence for the eval side of width, assembled honestly** [M]:

| comparison | arch family | metric | result |
|---|---|---|---|
| exp-0022 | classic 768×256 → 768×512 | fn 25k, n=1000 | **+16.0 ± 18.3** (1 doubling) |
| exp-0052 | classic uf1024 → uf2048 | fn 5k, n=300 | **+57.3 ± 17.7** (1 doubling) |
| exp-0058 | classic uf2048 → uf4096 | fn 5k, n=300 | **+15.1 ± 17.3** (1 doubling) |
| exp-0122 | SOTA 768 → 1024 | fn 25k, n=400 | **+6.1 ± 23.7** (0.415 doublings ⇒ +14.7/dbl) |
| exp-0120 | SOTA 768 → 1024 | **equal time**, n=300 | −8.1 ± 20.9 |
| exp-0149 | SOTA V-768/32 → S-1280/128 | fn 50k, n=200 | +34.9 ± 26.0 (0.74 dbl ⇒ +47/dbl) — **confounded**, neck also changes |

**The "+44 fn Elo per doubling" constant is a central estimate over a set that ranges
+14.7 to +57.3 per doubling, spans two different architectures, and has ±17–24 bars on
every member.** [D] It should be quoted as **+15 to +57**, not +44.

**An external replication at ~10× our sample size agrees, and adds the piece we were
missing.** Pawnocchio PR #68 (768 → 1024 accumulator, 16 → 32 input buckets) [S, §5.3]:
fixed nodes **+18.10 ± 5.64**, 10+0.1 **−11.02**, 40+0.4 −7.16, 40+0.4×4t −5.89,
**60+0.6×8t +5.02**. Same shape as ours — a better evaluator that loses at equal time —
and it does not turn positive until **long TC and eight threads**. ⇒ **The width
question is not "is width neutral" but "at what time control does it stop being
neutral", and our entire arch-testing regime (fn50k and 10+0.1, Threads 1–4) sits
strictly on the wrong side of that crossover.** That is a testing-protocol finding, not
a width finding, and it applies to every arch comparison in `ledger/experiments.jsonl`.

**And the field converged the other way.** Of thirteen surveyed top-20 engines (§5.4),
every threat-carrying one is at **768 or 1024**; only the four *threat-free* ones sit at
1536. Stockfish came down from 2560. **D48X at 1280 is on the wide side of the field for
a threat engine**, and the field's stated reason for coming down is that it wanted the
cycles for inputs and for the neck.

Against §3's **−49.5 Elo/doubling** of speed at 120 (or −28.9 at 70), the verdict is:

- **512**: −1.32 doublings ⇒ −58 fn Elo [E, at +44/dbl] against +54.8 of speed ⇒
  **a wash at 120, clearly negative at 70**. Also loses the A3-trunk verbatim warm
  transfer. **Reject.**
- **1280**: the incumbent. Its width premium is **fully paid for** — perf study 8
  derives the justified premium over SF18's 1024 as **1.080×**, not 1.25×, from SF18's
  own measured width-scaling instruction share (32.03%), with an independent
  derivation from our side agreeing to 0.3%. **The width is not why we are slow.**
- **3072**: −84.9 Elo of speed at 120 against +1.26 doublings ≈ +55 fn Elo ⇒
  **−30 net**. Also 331 MB of net, against a 15 GB box where RAM is the binding
  resource and a build beside a trainer has already caused a global OOM. **Reject.**

**Verdict: hold 1280. The width question is closed until R0 re-prices it.**

### 4.2 Feature set — HalfKAv2_hm ± FullThreats

**First, a correction that matters.** `searchopt/LEDGER.md` records "SF18's threat
feature space is LARGER than ours". [M, verified by me] That is true of the **`sf_18`
tag** (`FullThreats::Dimensions = 79,856`) and **false of our own constants reference**:
`git show eca43a97:src/nnue/features/full_threats.h:41` gives **`Dimensions = 60720`** —
*exactly ours*. Our feature set is a faithful transcription of `eca43a97`, and SF
*shrank* its threat space between the tag and master (it dropped the king as attacker
and as victim, and added pawn-push-or-attacks). Reference-fidelity is not a live issue
here.

**Second, the cost, corrected twice.** [M] Perf study 7 §5 refuted the "1.78× / half our
speed" figure with deterministic instruction counts on interleaved arms with verified
equal node totals:

| width | mix | instructions full/HalfKA-only |
|---|---|---:|
| 768 | wl50 | 1.247 |
| 768 | middlegame | 1.301 |
| **1280** | wl50 | **1.317** |
| **1280** | middlegame | **1.389** |

⇒ **~45–70 Elo of speed at 120/doubling, ~26–40 at 70.** The derived corollary "width
becomes 5.7× cheaper without the channel" came from the same refuted session; my model
puts the real figure at **1.81×** [D: marginal i/n per width unit, 2.513 with threats
vs 1.388 without].

**Third, a measurement of what the channel actually carries** [M, new — `net_anatomy.py`].
Weight mass in the HalfKA frame (threat weights are quantised at 64, HalfKA at 255, so
one threat weight is worth 255/64 = 3.984 HalfKA units):

| net | HalfKA \|w\| mean | threat \|w\| mean | E[\|contribution\|] threat share, linear model | random-sign √ model |
|---|---:|---:|---:|---:|
| D48X | 33.52 | 2.278 (=9.08 in-frame) | **52.0%** | 35.1% |
| D47V | 28.83 | 2.713 (=10.81) | 60.0% | 42.9% |

using the design maxima of ≤32 active HalfKA and ≤128 active threat features per
perspective (`design/11` §12.1). Three accumulator lanes have near-zero HalfKA mass and
four have near-zero threat mass, out of 1280 — **neither channel is vestigial anywhere**.
[Caveat: these are *upper-bound* active counts, not measured means; the true threat
share is lower than 52% and the √ model's 35% is the better central estimate.]

**Fourth, the eval value of the channel has still never been isolated.** The only
number is exp-0065's **+32.5 ± 17.3 fn** for the *whole* D40 architecture (HalfKAv2_hm
king buckets + FullThreats + the Reckless head shape) over classic features. The
"zero the threat rows and re-run the panels" measurement (QUIET 0.8811 → 0.7120) is an
upper bound on a net whose head was trained expecting a channel that is now identically
zero, and is explicitly *not* evidence about a net trained without it.

**The corrected EV of the threats-off bet.** [D] The staged `hka2048_probe.json`
(formerly "D49") was sized at "~+50 Elo vs V" from the refuted 1.78× and 5.7×. With the
corrected inputs:

| arm | speed Elo @120 | speed Elo @70 | eval cost | net @120 | net @70 |
|---|---:|---:|---:|---:|---:|
| threats OFF @1280 | +47.1 | +27.5 | −(0 … 32) | +15 … +47 | −5 … +28 |
| threats OFF @2048 | +16.0 | +9.3 | −(0 … 32) + width +44 | +28 … +60 | +21 … +53 |
| threats OFF @2560 | −2.0 | −1.1 | −(0 … 32) + width +57 | +23 … +55 | +24 … +56 |

**The bet survives the correction but its margin roughly halved, and it is now
sensitive to the width constant as well as the speed constant.** It is also the single
most expensive item in the study (a full training run, ~50–60 GPU-hours, plus a
10-line Pallas fix). **Ranked, but not first** — see §7.

**Fifth, and this is the finding that most changes my recommendation: the field has now
measured threats repeatedly and the answer is nearly unanimously yes.** [S, §5.4] Monty
PR #87 **+48.06 ± 3.3 STC**; Yukari +28.34 STC / +28.27 LTC; PlentyChess 7.0.2
**+45.49 fixed nodes** / −3.08 STC / +1.78 LTC; Stockfish 18 "up to 46 points";
Stormphrax 8.0.0 **+62.4 STC / +81.9 LTC**; Pawnocchio PR #64 +26.71 STC / +30.32 LTC.
Nine of the thirteen surveyed top-20 engines carry them, including all four above us.

**Every one of those results has the same signature we see in D48X: strongly positive at
fixed nodes, flat-to-negative at STC, positive at LTC.** So the correct reading of
"threats cost us 1.25–1.39× and their value is unmeasured" is not "so remove them"; it
is **"so we are testing them at the time control where nobody's threat result is
positive"**.

**Alternatives to removing threats.** Reckless (66,864 rows), Halogen (79,856), and the
59,808 + 4,560 pawn-pair family (SF, Stormphrax, Viridithas, PlentyChess, Pawnocchio)
all keep them; Integral, Obsidian, Alexandria and Astra have **none** and spend the
budget on width 1536 instead. **The split is clean: every threat engine is narrow, every
threat-free engine is wide, and they are interleaved in the ratings.** That is two
coherent design points, and D48X — threats *and* 1280 — sits between them.

⇒ **Revised recommendation vs the staged plan:** the isolated threats-off measurement is
still worth having, but it has dropped in priority and its bar should be set by the
field: if we run it, run it *at LTC as well as fixed nodes*, and expect it to lose.

### 4.3 Neck width — 8 / 16 / 32 / 128

**New measurement, with its null run first** (`neck_rank.py`, `neck_truncate.py`).
Statistic: participation ratio of the squared singular spectrum. Null: the same
statistic on i.i.d. matrices of identical shape with entries resampled from the
observed entries.

| net | L2 | L1 matrix [L2×ACC] obs / null | **L2 matrix [L2×L3] obs / null** |
|---|---:|---:|---:|
| d40-uho-v1 768/16 | 16 | 11.25 / 15.71 = 0.716 | **5.60 / 11.94 = 0.469** |
| **D48X 1280/16** | 16 | 14.41 / 15.81 = 0.911 | **9.95 / 11.83 = 0.841** |
| D47V 768/32 | 32 | 28.33 / 31.20 = 0.908 | **11.06 / 18.35 = 0.603** |
| d47s 1280/128 | 128 | 97.07 / 119.90 = 0.810 | **6.22 / 26.14 = 0.238** |

And the truncation-energy curves — what fraction of the neck→L3 map's Frobenius energy
lives in the top *k* directions:

| net | k=4 | k=8 | k=12 | k=16 | (null at k=8) |
|---|---:|---:|---:|---:|---:|
| D48X (L2=16) | 63.6% | 87.1% | 97.0% | 100% | 82.0% |
| D47V (L2=32) | 62.9% | 82.5% | 92.6% | 97.8% | 66.4% |
| d47s (L2=128) | **82.1%** | **95.7%** | **99.0%** | **99.8%** | 56.6% |

**Reading.** The map from the neck to L3 spans **6 to 11 effective directions in every
net we have trained**, at neck widths of 16, 32 and 128. At L2=128 eight directions
carry 95.7% of it against a null of 56.6% — a 128-wide neck is running at roughly
1/16 utilisation. At L2=16, obs and null are only 5 points apart at k=8, i.e. the
spectrum is barely more concentrated than noise, which is what you expect at or just
above the useful width.

**Caveats, because this is a weight-space statistic.** (i) It ignores the activation
covariance; a neck direction with a small singular value could still carry a
high-variance signal. (ii) Low rank in `l2_w` is equally consistent with "L3's 32 units
are redundant" as with "the neck is over-provisioned" — but *either* reading says
neither L2 nor L3 is the binding constraint. (iii) The four nets come from different
training runs; the pattern is consistent across them, which is the argument, but it is
not a controlled A/B.

**Verdicts.**
- **Widening the neck: reject.** 16 → 32 costs 4.9 Elo of speed at 120 and the
  measurement says the extra directions would not reach L3. This retires exp-0122's
  standing note ("L2=16 bottleneck now the leading suspect for width-saturation") —
  the suspect has an alibi.
- **Narrowing to 8: not worth it.** +2.5 Elo at 120 [D], and at L2=8 the AVX-512
  `l1_dot` still issues one `vpdpbusd` per group (a zmm holds 16 i32 lanes), so half
  the modelled saving is weight-traffic only. Real, tiny, and it spends eval headroom
  the rank data says is thin at 16.
- **The interesting move is not the width, it is what the tail does with it.** §4.4.

### 4.4 Depth and activation — the tail shape

**What SF did after we copied it** [M, verified directly at `eca43a97`]:

```
eca43a97:src/nnue/nnue_architecture.h:56-67
  fc_0      AffineTransformSparseInput<1024, 32>
  ac_sqr_0  SqrClippedReLU<32, WeightScaleBits+1>
  ac_0      ClippedReLU<32,   WeightScaleBits+1>
  fc_1      AffineTransform<64, 32>              // 32*2: the DUAL activation
  ac_sqr_1  SqrClippedReLU<32, WeightScaleBits>
  ac_1      ClippedReLU<32,   WeightScaleBits>
  fc_2      AffineTransform<32*2 + 32*2, 1>      // 128: BOTH layers reach the output
```

Three differences from ours, all in the same direction:
1. **Neck 32, not 15/16** (it was 15 at the `sf_18` tag; master doubled it).
2. **Dual activation** — every hidden unit contributes both `CReLU(x)` and
   `SqrCReLU(x)`, doubling the information each unit hands forward at the cost of one
   extra multiply per unit.
3. **A concat skip** — `fc_2` reads layer 0's dual activation *directly*, so a neck
   direction can reach the output without passing through `fc_1`'s CReLU.

Per-eval head MACs [D]: SF `eca43a97` = 32,768 + 2,048 + 128 = **34,944**; D48X =
20,480 + 544 + 1,280 = **22,304**. **SF spends 1.57× our head compute at 0.8× our
accumulator width.**

**Point (3) is the direct answer to §4.3's measurement.** Our neck's *only* path to the
output is one more linear layer plus a CReLU, and the rank data says that path uses ~10
directions. A dual activation plus a concat skip gives each neck direction three
independent routes to the output (`x`, `x²`, and through `fc_1`) without widening
anything.

**Cost of adopting it at our neck width** [D, `cost_model.py`]: keep L2=16 (so
`l1_dot`, the expensive part, is untouched), dual-activate the 16 → 32 values, run
`fc_1` 32 → 32, dual-activate to 64, and feed `fc_2` the 96-wide concat. Tail MACs
544 → 1,120. **+163 i/n = +2.29% of engine instructions = −3.9 Elo at 120, −2.3 at 70.**
Adopting SF's *full* shape (L2=32 as well) costs **+7.42% = −12.4 Elo at 120** and is
not recommended, because §4.3 says the width half of that change is the part that
would not pay.

This is the **best cost/benefit shape change in the study**: 2.3% of instructions, no
new feature space, no width change, and it is the shape the strongest engine in the
world converged to *after* the one we transcribed.

**It is also the best-*evidenced* item here, because someone else already measured it
standalone.** Stormphrax PR #239, dual concatenated activations alone: **fixed nodes
+6.95, STC +6.20, LTC +12.05** [S, §5.4]. Its skip-across-L2, measured separately:
**+4.45 fixed nodes, −2.66 STC (failed), +1.82 LTC (passed)**. Eleven of the thirteen
top-20 engines surveyed carry the dual activation; ARCHON, Reckless and Integral are
the exceptions. **Note the STC/LTC split on the skip half** — it argues for gating this
change at LTC or fixed nodes, not at 10+0.1, and for landing the dual activation first
and the skip second rather than as one patch.

**Reconciling this with §4.3's "do not widen the neck".** SFNNv13's commit message says
it doubled L2 from 16 to 32 *because* threat inputs had shrunk the accumulator and freed
cycles [S, §5.4] — and SF's neck feeds a **dual** activation, so 32 neck units hand 64
numbers forward and reach `fc_2` directly. Our 16 neck units hand 16 numbers into one
CReLU layer. **The two facts are consistent, and they combine into one rule: widen the
neck only *together with* the dual activation, never alone.** D47V is the natural
experiment for "L2=32 with a plain CReLU tail" and its neck→L3 map spans 11.06 effective
directions of 32 — the widening bought nothing because the tail had no route for it.
The pre-registered ladder that follows from this is: **(a) dual activation at L2=16
first; (b) only if (a) passes, L2=16→32 as a second, separately gated step** — which is
exactly Integral's decoupling discipline (§5.2, item 3).

**A cheap, decisive falsification exists and should be run first** — see R2 in §7:
a **frozen-trunk tail swap**. Take D48X's trained trunk, freeze the FT, retrain only
the ~170 k head parameters under (a) the current tail and (b) the dual-activation
tail, on the same data for the same steps. ~1 GPU-hour each. If (b) does not beat (a)
on heldout, the hypothesis is dead before a full run is spent.

### 4.5 Skip connections

**We already have two, and they are stronger than SF's.** [M]
- The **output-skip lane** (`sota_head.rs:430-433`): a learned linear read of all 1280
  activated pairwise features, per bucket, added in logit units. SF's equivalent at the
  `sf_18` tag is a *single* `fc_0` output lane; at `eca43a97` it is the difference of
  two. Ours is a full 1280-wide linear channel.
- The **PSQ1 skip** (`sota_head.rs:437`): an unclamped per-feature linear channel
  straight to centipawns, bypassing every nonlinearity — structurally SF's PSQT
  pathway.

**So "add skip connections" is already done at the *feature* end.** What is missing is
the *inter-layer* skip of §4.4 (layer 0's activations reaching the output). That is the
gap, and it is small and cheap.

One live improvement to the existing skip, already costed by perf study 8 §6.2:
folding the skip lane in as the `L2+1`-th row of the sparse matmul instead of a
separate dot would take it from 2.50% to ~0.15% of engine instructions. Study 8 §4.6
took the cheaper half of that (i8 + `vpdpbusd`, −1.89%); the rest needs a 17-lane
kernel. **Worth ~0.6% more, low priority.**

### 4.6 Output-head routing — the owner's queens/no-queens idea

**This is the only capacity axis in the architecture that is free at inference time.**
[D] Exactly one bucket's 22,353 parameters are touched per eval (§1.2). Doubling the
head bucket count doubles the `l1_w` table from 164 KB to 328 KB — still comfortably
resident — and changes the per-eval working set by **zero bytes**. Net file grows by
~180 KB on a 138 MB net.

**With one hard exception that has to be designed around.** The PSQ1 trailer is
`83,248 × buckets × i32` and its rows are touched **per feature event**, not per eval:
`psqt_add`/`psqt_sub` are ~3.0% of all engine instructions [M, perf study 8 §4.4], and
perf study 8 found the accumulate only vectorises because **`BUCKETS` i32 is exactly
32 bytes — one `ymm`**. Sixteen PSQ1 buckets would be two ymm and 10.7 MB of table.
**Therefore: decouple the head's bucket count from PSQ1's. Head buckets may grow;
PSQ1 stays at 8.** That decoupling is the enabling design move and it is not currently
possible — the loader validates one `out_buckets` field against both
(`sota_loader.rs`, `parse_psqt_section`).

**Is the queens/no-queens axis actually informative?** [M, new —
`bucket_occupancy.py`] P(any queen on board | bucket), over four real ARCHON position
distributions:

| distribution | n | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plateau (unconverted wins) | 308 | 0.0 | 22.2 | 6.1 | **35.1** | **47.9** | 78.6 | 92.2 | 100 |
| eval bench | 700 | 10.5 | 19.1 | 28.7 | **49.2** | **62.2** | 82.8 | 91.2 | 100 |
| loss corpus | 135 | 0.0 | 30.0 | 39.1 | **62.5** | 96.7 | 100 | 100 | 100 |
| scale-check 800 | 800 | 9.7 | 20.4 | 32.1 | **45.5** | 58.4 | 84.7 | 97.0 | 100 |

and the bucket occupancy of the positions where we actually lose Elo:

| distribution | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | effective buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plateau (unconverted wins) | 0.0 | 2.9 | 10.7 | 12.0 | **30.5** | 22.7 | 16.6 | 4.5 | 5.67 / 8 |
| loss corpus | 3.7 | 14.8 | 17.0 | **17.8** | **22.2** | 13.3 | 7.4 | 3.7 | 6.90 / 8 |

**The queen axis is collinear with piece count at the extremes and near-orthogonal in
buckets 3–5 — which is exactly where our unconverted wins live** (buckets 3+4 hold
42.5% of plateau positions and 40.0% of the loss corpus, and the queen split there is
35/65 to 49/51). A queenless 20-piece middlegame and a 20-piece position with two queens
are different games sharing one head today.

**Caveat, stated plainly:** the net already *sees* the queens through the feature
transformer. Bucketing does not add information; it adds **specialised head capacity**
and removes the need for one 16-dim neck to serve both regimes. The argument is
capacity allocation, not information — and the cost is data dilution per bucket, which
is the real risk on our data budget.

**A zero-training-cost falsification exists** (R3 in §7): split the *existing* heldout
by (bucket, queens-present) and test whether D48X's residual error differs
systematically within a bucket. If the residuals are indistinguishable, the split has
nothing to specialise on and the idea dies for free.

### 4.7 Dual big/small nets

**SF deleted it.** [M, verified by me at `eca43a97`, 2026-07-06 — our own constants
reference — where `git grep -il "smallnet\|NetworkSmall\|SmallFeatureTransformer" -- src/`
returns nothing, and `evaluate.cpp` has no `simple_eval`, no 962 threshold, and no 277
re-evaluation fallback. The same grep at `sf_18` (2026-01-16) returns three files.]

**And it was small even while it existed** [M, perf study 7 §4, callgrind call counts on
an AVX2 rebuild proven node-identical to the official binary]: the small net handled
14.4% of evals on wl50 and 24.0% on default FENs, at ~6× cheaper per eval; forcing
SF18 big-always raises its instruction total only **+3.65% / +6.50%**.

**For us the ceiling is better but still not large.** Our expensive tier is dearer than
SF's, so a threat-free cheap tier firing on the same 14–24% of nodes saves
proportionally more: perf study 7 §7 puts the honest ceiling at **+4–8% NPS ⇒ +7 to
+14 Elo at 120, +4 to +8 at 70**, for "a trained threat-free head or small net, a
material predicate, accumulator plumbing, and a fallback band" — weeks of work, with a
real eval-consistency risk (two evals of the same position must not disagree across a
TT boundary).

**And the removal commit says why** [S, `official-stockfish/Stockfish` `ab8f901d`,
2026-05-04]: both non-regression SPRTs **failed** and it was merged anyway —
*"multiple net architectures to maintain, multiple nets to train … a pure slowdown of
that magnitude would lead to more dramatic losses, suggesting that the small net's lower
eval quality is deleterious."* i.e. the small net was costing eval quality that its
speed did not repay, and the maintenance burden decided the rest.

**Verdict: reject.** Not because the idea is bad — adaptive eval cost is a genuine
conditioning axis and `searchopt/LEDGER.md` is right to flag it — but because the
strongest engine tried it, measured it as eval-negative, and walked away; the measured
ceiling for us is ~10 Elo; and the cost is among the highest in this study. Multi-net
now survives only as value+policy in MCTS engines. If it ever returns, run the free
probe first: log `|simple_eval| > 962` frequency in our own trees and check our node mix
matches SF's.

### 4.8 Quantisation and format

The current scheme [M, `sota_head.rs:225-232`, `design/11` §12.6]: HalfKA i16 at
QA=255, threats **i8 at QT=64**, L1 i8 at QB=64, `FT_SHIFT=9`, L2/L3 f32,
`network_scale` in the header. This is Reckless's and Integral's shape, independently
converged (both use QA=255, `FT_SHIFT=9`; Integral uses QB=128).

**The split frame is the standing wart and it is now mostly paid.** Because the two
transformers live in different frames, `activate` had to compute `round(t · 255/64)`
per lane — 24 of the merge loop's 48 instructions, ~2.5% of the engine, against SF's
single `vec_add_16`. Perf study 8 §4.5 fixed it by carrying the ratio as **4** instead
of 255/64 (a 0.39% gain change, measured at **0.14 SF cp mean / 2.42 max** over 800
positions), for **−2.43%**. Rebasing the stored weights to a single frame — the
"proper" fix — was measured and **rejected on range**: the trained net reaches |w| = 126
where only |w| ≤ 31 survives, so 0.031% of weights saturate and the tail reaches
**205 SF cp**. [M]

**So the i8 threat frame is not a mistake; it is a footprint decision.** Perf study 8
§7.11 nails it: storing the threat rows pre-widened at i16 is bit-exact and −2.10%
instructions, and **still loses** because the table goes 78 MB → 155 MB against a 32 MB
L3 (cache-misses ×1.299, cycles ×1.078). **Do not "fix" the quantisation asymmetry by
widening the table.**

**One open format item worth carrying into any retrain** [D]: if a net is trained with
the threat transformer's output scale constrained so its weights fit the 255 frame
without saturation, the merge becomes a bare add and ~2.5% comes back for free. That is
a *trainer-side regularisation* question (weight-norm penalty on the threat FT), not an
engine change, and it is the one quantisation idea in this study that has never been
tried.

### 4.9 The eval-scale constant — the field's biggest reported win, already closed here

The single largest Elo number the survey turned up is not an architecture change at all.
[S] cosmobobak trained 11 nets at λ = 0.0 … 1.0 and found that **higher λ monotonically
inflates the net's output scale**; feeding that to a search whose constants were tuned at
a different scale collapses strength. Changing only the inference `SCALE` constant
(400 → 229) recovered **+28 Elo**, and Viridithas shipped `SCALE = 240` for
**+46.12 ± 4.63 at 60+0.6**. Halogen went further and made its scale an SPSA parameter
(`TUNEABLE_CONSTANT float SCALE_FACTOR = 192.5f`).

**This is exactly the symptom `engines/canonical/MANIFEST.md` records** — a stronger eval
the search cannot exploit — and `network_scale` is a single i32 in our CENN header with
`tools/nnue/rescale_net_header.py` already able to change it without retraining. So it
demands an answer.

**The answer is that we have already tested it, twice, and it lost in both directions.**
[M] `searchopt/LEDGER.md`: *"`EVAL_SCALE` tests the frame — it is a global
pruning-aggressiveness knob; **−27.8 ± 16.8 over 188 games**"*, and
`docs/research/eval-frame-scale-model-2026-07-27.md` §11.4 *"killed the blanket rescale
in **both directions** and concluded the constants sit at a joint optimum"*.

**And the reconciliation is that we already do the thing they were paid for.** Our
pipeline calibrates `network_scale` per net as a hard pre-game gate
(`tools/nnue/calibrate_network_scale.sh`, `net_scale_check.sh`, D46/CLAUDE.md); exp-0160
records "calibration is clean (scale 419, ratio 0.992)". Viridithas's +46 came from
shipping an *uncalibrated* 400 against a net whose λ had moved the scale. **That is a
bug-class D46 closed for us before it could cost us anything**, and it is worth writing
down as such rather than re-opening.

**One residual worth naming, though.** Our calibration targets the **label** frame — it
makes the eval agree with the training targets. cosmobobak's mechanism is about the
**search-optimal** frame, which need not be the same number. The `EVAL_SCALE` experiment
is the right test of that distinction and it came back negative in both directions at
n=188, which is a thin sample for a ±17 bar. If the box is ever idle, a proper 3-point
scale sweep at n ≥ 1,000 would retire it for good; it is not worth displacing anything in
§7 for.

### 4.10 Policy and auxiliary heads

**Policy: stay ditched, and the reasons have not changed.** [M]
- exp-0121: co-training a best-move policy aux at λ=0.001, deployed eval-only, measured
  **−14.0 ± 23.0** on the eval at equal time. "The zero-NPS aux steals trunk capacity."
- The sb30 ON/OFF ablation (300 games each side): identical scores, **−381.7 ± 29.5 vs
  −381.7 ± 29.2** — a ~17%-top-1 deployed policy is worth exactly zero.
- The owner's bar is 35% top-1; the best we have reached is 27.2% (S-post), and the
  squeeze is dose-responsive (V went 18.7 → 13.7).

If it returns, the only defensible form is the roadmap's: a **frozen-trunk head
finetune** on policy-specialised data, never a joint objective sharing trunk gradient.

**The uncertainty head is a better bet and is already scaffolded.** `l3_var_weights` /
`l3_var_biases` exist in `SotaHead`, load from a sidecar (so the CENN is unchanged),
cost ~32 extra FLOPs, and return `None` for every net that lacks them — meaning a
variance-less net is bit-identical to today. Per `design/17` §3.6 an auxiliary head
that only *steers* search "cannot corrupt correctness, only spend". Given the measured
defect is under-resolution in balanced positions, a per-position σ̂ that widens the
aspiration window and softens RFP exactly where the eval is unsure is well matched to
it. **Cheap, low-risk, untested.**

The WDL head (`l3_wdl_weights`) is similarly scaffolded and would let the outer shape
use a real win probability instead of a cp proxy — relevant because `searchopt`'s own
standing rule is that centipawns are not additive into outcomes.

**★ And the consumer side is already built and currently inert** [M, measured in
`searchopt` on 2026-07-28 and reported to this study]. `src/search/mod.rs` carries two
ARCHON-original time-management soft-stop factors, both defaulting off:

- **`ARCHON_UNC_TM_C2`** — `1 + c2·(σ̂ − σ̄)/σ̄`, clamped [0.80, 1.25] ("unclear ⇒ think
  longer"), `src/search/mod.rs` ~line 236.
- **`ARCHON_WDL_TM_C3`** — `1 − c3·decisiveness`, clamped [0.80, 1.00],
  `wdl_tm_factor` at `src/search/mod.rs:250`, reading `eval_ctx.last_wdl()` at ~line
  2433 as `|P(win) − P(loss)|`.

Verified inert on the canonical net: with `ARCHON_TM_TRACE=1` the soft-stop factor
sequences are **byte-identical** across baseline, `ARCHON_WDL_TM_C3=0.5` and
`ARCHON_UNC_TM_C2=0.5` — 17 trace rows each, `opt=175` throughout. D48X carries neither
head, so `last_wdl()` returns `None`, `root_sigma` stays 0.0, and both factors return
exactly 1.0 by their own guard.

**Why this changes the economics:** pricing a WDL or uncertainty sidecar normally means
paying for the head *and* writing the consumer. Here the consumer costs **zero
engineer-days**. That is a materially different case from every other aux-head proposal
in `design/17`.

**Two caveats that stop it becoming an overclaim, both supplied with the measurement:**
(i) the Elo value of those TM factors is **unmeasured** — they have never run on a net
that feeds them, so "consumer exists" is not "consumer pays"; (ii) there is a
**head-free substitute** — the same factors can be driven from the search's own root
score, and a sidecar has to beat that cheaper alternative, not merely beat nothing.
**The head-free arm is therefore the mandatory control, and it should be run first**
because it costs no GPU at all.

**The motivating asymmetry, also measured** [M, `searchopt/LEDGER.md`, "TIME
ALLOCATION"]: at 10+0.1 we spend **18% less of our own time budget than SF18 in balanced
positions** (|eval| < 0.25, conditioned ex ante on the previous own-eval) and **33% less
when clearly winning**, while the conversion band itself is at parity. **"Clearly
winning" is exactly the regime where we convert 12% of won books to SF's 84%** (§1.4) —
we under-think precisely where we fail to convert. A decisiveness signal that made us
think *longer* rather than shorter when ahead is pointed straight at the measured defect.
Note the sign: `ARCHON_WDL_TM_C3` as written *shortens* the search when decisive
(`1 − c3·decisiveness`), which is the opposite of what this asymmetry suggests — so the
first thing to establish is the sign, not the head.

---

## 5. WHAT THE REFERENCE ENGINES SAY

### 5.1 Stockfish

Two revisions, and the difference between them is the most useful thing in this section.

| | `sf_18` tag (2026-01-16) — what we transcribed the head from | **`eca43a97` (2026-07-06) — our constants reference** |
|---|---|---|
| nets | 2 (big 1024 + small 128), gated on \|simple_eval\| > 962 with a \|nnue\| < 277 big-net fallback | **1** |
| threats | FullThreats **79,856** | FullThreats **60,720** — *identical to ours* |
| L2 | 15 (+1 skip lane) | **32** |
| tail | fc_1 30→32, fc_2 32→1 | **fc_1 64→32, fc_2 128→1 (both layers concat)** |
| activations | SqrCReLU + CReLU concat at layer 0 only | **dual activation at BOTH layers** |
| output buckets | 8, `(count-1)/4` | 8, `(count-1)/4` |
| Finny cache | 64 king squares × 2 colours, big + small (~320 KiB/thread) | one cache (~272 KiB/thread) |
| threat refresh | own king crosses the d/e file only (strict subset of HalfKA refreshes) | shares the HalfKA trigger |

Citations: `sf_18:src/evaluate.cpp:43-90`, `sf_18:src/nnue/nnue_architecture.h:39-71`,
`eca43a97:src/nnue/nnue_architecture.h:43-45,56-67`, `eca43a97:src/evaluate.cpp:42-69`,
`eca43a97:src/nnue/features/full_threats.h:41`.

Outer shape at `eca43a97` (unchanged from `sf_18` in every constant):
```
Value nnue = psqt + positional;
int nnueComplexity = |psqt − positional|;
optimism += optimism * nnueComplexity / 476;
nnue     -= nnue     * nnueComplexity / 18236;
int material = 534 * count<PAWN>() + non_pawn_material();
v = (nnue*(77871+material) + optimism*(7191+material)) / 77871;
v -= v * rule50_count() / 199;
```
**We implement every line of this except the two complexity terms** — see §7 R1.

### 5.2 Reckless and Integral

*(Rating note: the tracked notes quote Reckless at ~3833 SPCC "#2 at study time" and give
Integral no number. The independent CCRL 40/15 4-CPU list of 2026-07-23 puts them at
**3646** and **3618** — different list, different scale, same ordering. Use the CCRL
column of §5.4 for any cross-engine comparison; the SPCC figure is not commensurable
with it.)*

⚠️ **Neither clone is on disk.** `reckless_reference/Reckless/` and the whole
`integral_reference/` tree are absent; the Integral notes were **deleted in commit
`1f8c111`** ("canonical: promote D48X…"), apparently as collateral, and had to be
recovered with `git show 1f8c111^:integral_reference/<file>`. **Everything in this
subsection is second-hand from the tracked notes and could not be verified against
source. `integral_reference/` should be restored.**

| | Reckless v0.10.0-dev | Integral `3f15ba6b` | **ARCHON D48X** |
|---|---|---|---|
| accumulator | 768 | **1536** | 1280 |
| king buckets | 10 (8 of them on ranks 1–2) | 12 (+ training factorizer) | 32 |
| threat features | **66,864 rows, 51.4 MB i8** | **none** | 60,720, 77.7 MB i8 |
| tail | 768→16→32→1 | 1536→16→32→1 | 1280→16→32→1 + 1280-wide skip + PSQ1 |
| output buckets | 8, non-uniform | 8, uniform `(pc−2)/4` | 8, non-uniform |
| QA / QB / FT_SHIFT | 255 / 64 / 9 | 255 / **128** / 9 | 255 / 64 / 9 |
| net size | 63.3 MB | 28.5 MB | 138.3 MB |

**Three observations that bear on our decisions.**
1. **Integral reaches top-6 with zero threat features** and spends the budget on width
   and king buckets. Reckless reaches #2 *with* them at 51.4 MB. The field does not
   agree, which is why §4.2 wants the isolated measurement rather than an argument.
2. **The one thing Integral does that neither we nor Reckless do is a training-time
   king-bucket factorizer** (shared virtual row folded in before export, zero inference
   cost). We tested this: exp-0099, **+6.9 ± 18.0** at fixed nodes, n=300 — a null,
   but an underpowered one at ±18 for an effect plausibly worth +10.
3. **Integral deliberately decoupled depth and width**: when it added the L2/L3 tail it
   *shrank* L1 from 1536 to 1280 for three months to keep SPRT attribution clean. That
   is the discipline §7's R2 should follow.

Our own external anchors [M, exp-0060/0062/0063, 100 games each, 10+0.1]: SF16 −78 ± 18,
SF18 −191 ± 23, **Integral v6 −70 ± 22** — i.e. at that time Integral, with a
threat-free 28.5 MB net, was beating us by about as much as SF16 was.

### 5.3 Pawnocchio — found, and it is the most relevant engine in the field

**https://github.com/JonathanHallstrom/pawnocchio** — UCI engine in **Zig**, GPLv3, by
Jonathan Hallström; nets in a separate repo. **CCRL 40/15 4-CPU: 3643, #4** (list of
2026-07-23). [S — read from source on GitHub by the survey agent and re-derived
arithmetically; not verifiable in this repo, which has no clone.]

| | Pawnocchio `main` (v2.0.1) | **ARCHON D48X** |
|---|---|---|
| PSQ features | 768 × **16** king buckets, mirrored = 12,288, **i16** | HalfKAv2_hm 22,528, 32 buckets, i16 |
| threats | **59,808**, i8, no king bucket | 60,720, i8, no king bucket |
| pawn pairs | **4,560** (unordered pairs of 96 pawn ids, active within ±1 file), i8 | **none** |
| accumulator | **768** | 1280 |
| activation | clamp[0,255], pairwise mul, `>>9` | identical |
| tail | 16 → **dual (CReLU ‖ sqr)** 32 → 32 → 1 | 16 → CReLU → 32 → 1 |
| tail arithmetic | **all i32 — no floats anywhere** | f32 |
| output buckets | 8, `(popcount−2)/4` | 8, non-uniform table |
| quant | QA 255 / **QB 128** / Q 64 / SCALE 400 | QA 255 / QB 64 / Q 64 / SCALE 400 |
| net | 68.4 MB, ~59.0 M params | 138.3 MB, 107.4 M params |

Its `dev` branch (2026-07-13) has already moved to **L1 1024, 32 input buckets, L2 32**,
plus a hard-coded 512-entry L1 neuron-pair permutation for sparsity — i.e. converging on
Stockfish's current shape.

**The single most useful thing it gives us is an independent replication of our own
width result at ~10× the sample size.** PR #68 (768 → 1024 accumulator, 16 → 32 input
buckets):

| condition | Elo |
|---|---:|
| fixed nodes | **+18.10 ± 5.64** |
| 10+0.1 | **−11.02** |
| 40+0.4 | −7.16 |
| 40+0.4 × 4 threads | −5.89 |
| **60+0.6 × 8 threads** | **+5.02** |

**Better evaluator, worse at equal time, and it does not turn positive until long TC and
many threads.** That is the D48X promotion ruling, measured by someone else, on a
different engine, with tight bars — and it says the ruling's *falsifiable prediction* is
right in shape but that the thing which converts a better eval is **time control and
thread count**, not only search quality. **Our own re-test of `engines/runs/d48x-vs-v`
should therefore include an LTC arm, not just a 10+0.1 one.**

### 5.4 The field, 2025–26 — what converged and what did not

[S — assembled by the survey agent from primary sources on GitHub and CCRL; cross-checked
against struct arithmetic per engine. Not verifiable inside this repo. The Stockfish rows
*are* verifiable and were checked by me against the local clone up to `origin/master`
@ 2026-07-16; anything later is web-only and flagged.]

| engine | CCRL 40/15 4CPU | acc/persp | threats? | tail | out buckets |
|---|---:|---:|---|---|---:|
| Stockfish | 3649 | 1024 | **yes** 59,808 + 4,560 PP | 32 → dual 64 → 32 → dual 64 → 1 **+ skip** | 8 |
| Reckless | 3646 | 768 | **yes** 66,864 | 16 → 32 → 1 | 8 |
| PlentyChess | 3644 | 1024 | **yes** 59,808 + 4,560 | 16 → dual 32 → 32 → 1 **+ skip** | 8 |
| **Pawnocchio** | 3643 | 768 | **yes** 59,808 + 4,560 | 16 → dual 32 → 32 → 1 | 8 |
| Obsidian | 3637 | **1536** | no | 16 → dual 32 → 32 → 1 | 8 |
| Alexandria | 3636 | **1536** | no | 16 → dual 32 → 32 → 1 | 8 |
| Stormphrax | 3635 | 1024 | **yes** 59,808 + 4,560 | 32 → dual 64 → 64 → 1 **+ skip** | 8 |
| Viridithas | 3632 | 1024 | **yes** 59,808 + 4,560 | 32 → HardSwish → 64 → **SwiGLU + skip** → 32 → 1 | 8 |
| Astra | 3631 | **1536** | no | 16 → dual 32 → 32 → 1 | 8 |
| Halogen | 3625 | 768 | **yes** 79,856 | 16 → dual 32 → 32 → 1 | 8 |
| Caissa | 3622 | 1024 | no | **SCReLU → 1, single layer** | 8 |
| Clover | 3620 | 1280 | no | **SCReLU → 1, single layer** | 8 |
| Integral | 3618 | **1536** | no | 16 → 32 → 1 | 8 |
| **ARCHON D48X** | — | **1280** | **yes** 60,720 | **16 → 32 → 1** | 8 |

**Five conclusions, and the last one is uncomfortable.**

1. **Width converged DOWN to 768–1024, not up.** Every threat-carrying engine is at 768
   or 1024; only the four *threat-free* engines sit at 1536. Stockfish came down from
   2560 (SFNNv8). SFNNv13's own commit states the mechanism: *"Since the introduction of
   Threat Inputs … the accumulator/L1 size has significantly decreased. This has made it
   more computationally favorable to double the L2 size from 16 to 32."* **The field
   spent its width budget on inputs and then moved the freed cycles into the neck.**
   D48X at 1280 is inside the field but on the wide side of it for a threat engine.

2. **Threat inputs swept the field in one year, with big measured numbers.** Monty
   PR #87 **+48.06 ± 3.3 STC**; Yukari +28.34 STC / +28.27 LTC; PlentyChess 7.0.2
   **+45.49 fixed nodes**, −3.08 STC, +1.78 LTC; Stockfish 18 "up to 46 points";
   Stormphrax 8.0.0 **+62.4 STC / +81.9 LTC**; Pawnocchio PR #64 +26.71 STC / +30.32 LTC.
   **Every one of them shows the same signature: strongly positive at fixed nodes,
   flat-to-negative at STC, positive at LTC.** That is D48X's signature exactly.
   ⇒ **This materially lowers my confidence in R7 (remove threats).** The field's answer
   to "are threats worth their cost" is a nearly unanimous yes, measured with far more
   games than we have.

3. **Dual concatenated activations (CReLU ‖ SqrCReLU) are now standard and we do not
   have them.** Eleven of the thirteen engines above have them; ARCHON, Reckless and
   Integral do not. **Stormphrax measured the change standalone (PR #239): fixed nodes
   +6.95, STC +6.20, LTC +12.05.** That is a directly transplantable measured number for
   R2, and it is the best-evidenced single item in this study.

4. **Skip connections are real and independently rediscovered by four authors in one
   year** — SF (`fc_2` reads L1's activations plus a two-neuron scalar bypass),
   Stormphrax (`kSkipL2`), PlentyChess, Viridithas. **Stormphrax measured its skip
   standalone: fixed nodes +4.45, STC −2.66 (failed), LTC +1.82 (passed).** Smaller than
   the dual activation, and STC-negative — which matters for how we gate it.

5. **Two engines are still top-20 on a single-layer SCReLU net with no pairwise
   multiply** (Caissa 3622, Clover 3620). Depth is not obviously load-bearing. Read that
   as a caution against over-investing in the tail, not as a refutation of item 3 — the
   measured deltas in item 3 are from controlled A/Bs and this is a cross-engine
   observation.

**One item the field measured that we have already done:** i8 quantisation of the threat
weight rows. PlentyChess measured **+10 Elo STC / +3 LTC for only −2 fixed nodes**, and
it took Stockfish from −9 STC to neutral (+15 on ARM) — it is a *memory-bandwidth* fix,
not an arithmetic one. **We already store threats at i8** (`THREAT_QUANT = 64`), and perf
study 8 §7.11 independently rediscovered the mechanism from the other side by measuring
that widening them to i16 *loses on cache*. §4.8's reading stands and is now corroborated
by an external A/B.

**One item the field measured that we should copy:** SFNNv11 **pruned high-churn threat
features** (piece→king threats), "saving 13MB of net space and approximately 0.8 threat
feature updates per incremental accumulator update", non-regression. We already exclude
the king as attacker *and* as victim (`sota_features.rs:MAP`, all −1 in the king row and
column) — so that specific prune is already ours. The *general* move is not: see R8 in §7.

**And one thing the field says is bad, so we do not have to find out:** x-ray threats and
psq-suppression schemes measured **">50 Elo fixed-nodes loss in Monty"**.

---

## 6. THE OWNER'S "512 → 8 → 8 AND BE SUPER FAST", ANSWERED

The instinct is right about *where* to look — per-node cost is our binding constraint —
and wrong about *what is expensive*.

**Decomposition** [D, `cost_model.py`]:

| step | i/n | speed | Elo @120 | Elo @70 |
|---|---:|---:|---:|---:|
| D48X 1280/16/32 | 7,117 | — | — | — |
| → 512/16/32 (**width only**) | 5,187 | 1.372 | **+54.8** | +31.9 |
| → 512/8/8 (**head only, on top**) | 5,041 | 1.029 | **+4.9** | +2.9 |
| (for contrast) D48X → 1280/8/8, width kept | 6,954 | 1.023 | +4.0 | +2.3 |

**92% of the speed comes from the width, 8% from the head.** The reason is §1.3: the
entire post-`l1_dot` tail — `l1_dequant` + `l2_tail` + `l3` + `skip_dot` — is **4.70%
of engine instructions**. You cannot get "super fast" out of 4.7%.

**And the width half is the trade the project has already measured to be a wash.**
1280 → 512 is −1.32 doublings; at the standing +44 fn Elo/doubling that is **−58 fn
Elo** against **+54.8** of speed at 120 ⇒ **−3 net**, and at 70 Elo/doubling it is
**−58 + 31.9 = −26**. It also throws away the verbatim A3-trunk warm transfer, which
is why 1280 was picked over 1536 in the first place.

**The one part worth keeping from the idea is the neck**, and §4.3's rank measurement
says 8 is *plausible* (the neck→L3 map spans ~10 directions at every width tested) but
worth only +2.5 Elo at 120 — while spending eval headroom that at L2=16 is already thin
relative to its null. **Not recommended, but it is the only part of the proposal that is
not refuted.**

**If the goal behind the proposal is "make eval cheap enough that the search can
afford more nodes", the arithmetic points somewhere else entirely:** removing the
*threat* channel at unchanged width buys +47 Elo of speed at 120 (§4.2), and perf study
8's four named implementation items (`at_full` replay 5.7%, threat-system duplication
~1.3%, make/unmake 6.1%, accumulator memmove 2.0% = **~15% of the engine**) buy roughly
**+26 Elo at 120 / +15 at 70 with zero eval risk at all.**

---

## 7. RANKED RECOMMENDATIONS

Ranked by (expected Elo) × P(survives) ÷ cost. "Cost" is engineer-days + GPU-hours;
a full ARCHON training run is ~50–60 GPU-hours at the measured 178k pos/s and 2.3–3.3
min/superbatch. Every entry carries a **pre-registered kill test** chosen so it can
fail cheaply and *before* the expensive part.

### R0 (GATE) — Re-measure Elo per doubling of speed

- **Why first:** §2. The constant that prices every other row has a 1σ bar of ±~95 and
  a 95% CI that includes zero. Three of the rows below change rank if it is 70 rather
  than 120.
- **Expected Elo:** 0 directly. **Decision value: it re-prices the entire study.**
- **Cost:** 0 engineer-days, 0 GPU, ~1 box-day of games.
- **Pre-registered test:** same PGO binary, same net, `-each nodes=N` vs
  `-each nodes=2N`, n ≥ 2,000, UHO book, colour-paired. Elo/doubling = the measured
  difference. (The nodes form removes time-management confounds; run the 10+0.1 vs
  5+0.05 wall-clock form as a second arm if the box is idle.) **Report with its CI and
  pin it in `searchopt/LEDGER.md`; every future arch decision cites the pin.**
- **Risk:** none. **Score: gate.**

**R0b — the same gate applies to the TEST CONDITION, and this one is new.** Pawnocchio
PR #68 [S, §5.3] measured a width increase at five conditions: **+18.10 ± 5.64 fixed
nodes, −11.02 at 10+0.1, −7.16 at 40+0.4, −5.89 at 40+0.4×4t, +5.02 at 60+0.6×8t.** Every
threat-feature result in §5.4 has the same shape. **Our entire architecture-testing
regime — fn50k and 10+0.1 at Threads 1–4 — sits strictly on the side of that crossover
where a better evaluator reads as a loss.** Combined with `searchopt/LEDGER.md`'s finding
that n=406 gives a ±40–50 Elo observed half-width, this means **several of exp-0149…0163's
architecture verdicts are underpowered *and* measured in the wrong regime.**
**Recommendation: every arch gate from here carries an LTC arm** (40+0.4 or better,
multi-thread if available), and D48X-vs-D47V should be re-run with one before its
falsifiable prediction (§1.5) is judged. Cost: box time only.

### R1 — Wire up the complexity channel (the unconsumed output)

- **What:** we already compute the two halves of SF's `nnueComplexity` and throw the
  split away. `sota_head.rs:437` returns `value = (l3 · scale) + psqt_delta`; the
  psqt and positional channels are separately available one line earlier.
  `src/eval/mod.rs:325` says "complexity = 0 and the complexity blend still vanishes" —
  a comment written when the head had no PSQT pathway, which stopped being true when
  T6c/PSQ1 landed.
- **Add:** `complexity = |psqt_cp − positional_cp|`; `optimism += optimism·complexity/476`;
  `nnue -= nnue·complexity/18236` — SF's constants, unchanged at `eca43a97`
  (`eca43a97:src/evaluate.cpp:55-57`) and therefore inside our transcription contract.
- **Why it is well matched to our defect:** the term feeds `optimism`, which drives the
  aspiration window. `searchopt/LEDGER.md` measures our aspiration window collapsing to
  the bare base of 5 in exactly the balanced band where we churn 1.561× SF, and the
  `asp30` arm's targeted 12.3% churn reduction in that band is already under SPRT.
  Complexity is the principled version of the same lever.
- **Expected Elo:** +3 to +10 [E]. **P(survives): 0.35.**
- **Cost: 0.5 engineer-days, 0 GPU, 0 retraining.** ~3 instructions per eval.
- **Pre-registered kill test:** fn50k, n=600, identical binary and net on both sides
  (the change is search-side only). Bar: land if the point estimate clears +5 with the
  CI excluding 0; kill otherwise. **Null panel required first**: run the same 600-game
  harness with the term computed and multiplied by zero, to confirm the harness reads
  0.0 ± bar.
- **Score: highest.** It is the cheapest item in the study and it is not an architecture
  change at all — it is an output the architecture already produces and the engine
  discards.

### R2 — Adopt SF `eca43a97`'s tail: dual activation + inter-layer concat skip

- **What:** §4.4. Keep L1=1280 and L2=16 (so `l1_dot` is untouched). Dual-activate the
  neck (16 → 32), `fc_1` 32 → 32, dual-activate (→ 64), `fc_2` reads the 96-wide
  concat of both layers.
- **Why:** §4.3 measures the neck→L3 map spanning 6–11 effective directions in every
  net we own. The tail cannot use more neck than it has routes for. This adds routes
  instead of width, and it is what SF adopted after we copied its predecessor.
- **Cost in speed:** +2.29% instructions = **−3.9 Elo at 120, −2.3 at 70** [D].
- **Expected Elo:** **+10 to +25 net of the speed cost**, and this is now the
  best-evidenced item in the study rather than an estimate: **Stormphrax measured the
  dual activation standalone at +6.95 fixed nodes / +6.20 STC / +12.05 LTC** [S, §5.4],
  eleven of thirteen surveyed top-20 engines carry it, and ARCHON is one of the three
  that do not. **P(survives): 0.60** (raised from 0.50 on the external replication).
- **Cost:** ~4 engineer-days (engine kernel + loader + trainer + CENN dim field) +
  **1 GPU-hour for the kill test** + 50–60 GPU-hours only if the kill test passes.
- **Pre-registered kill test (run BEFORE any full run): the frozen-trunk tail swap.**
  Freeze D48X's FT; retrain only the ~170 k head parameters, same data, same steps, two
  arms — current tail vs dual-activation tail. ~1 GPU-hour each. **Bar: kill unless the
  new tail's heldout improves by ≥1.0% relative.** A tail that cannot beat the old tail
  on a *fixed* trunk will not beat it on a co-trained one.
- **Discipline note:** follow Integral's precedent and change **depth only**, holding
  L1 and L2 fixed, so the SPRT attributes cleanly.
- **Risk:** medium. The dual activation needs new quantisation care (SF shifts by
  `WeightScaleBits+1` at layer 0 and `WeightScaleBits` at layer 1 — the shifts are
  per-layer, `eca43a97:src/nnue/layers/sqr_clipped_relu.h:34`), and our tail is f32
  where SF's is int8, so the transcription is of *shape*, not of quantisation.

### R3 — Decouple head buckets from PSQ1 buckets, then split buckets 3–5 on queens

- **What:** §4.6. Add a separate `head_buckets` field to CENN-v2 (PSQ1 keeps
  `out_buckets` = 8, because 8 i32 is exactly one `ymm` and perf study 8 §4.4 shows the
  vectorisation depends on it). Then route the head on `(piece-count bucket, any queen
  on board)` for buckets 3–5 only: 8 → 11 heads.
- **Why:** it is the only free-at-inference capacity axis we have (§4.6), the queen
  split is near-orthogonal to piece count exactly in buckets 3–5, and buckets 3–4 hold
  42.5% of unconverted-win positions and 40.0% of the loss corpus [M].
- **Expected Elo:** +5 to +15 [E]. **P(survives): 0.40** (the risk is data dilution).
- **Cost:** ~2 engineer-days + 50–60 GPU-hours for the retrain.
- **Pre-registered kill test — ZERO training cost:** take the existing heldout set and
  D48X as it stands; split each piece-count bucket by queen-presence and compute the
  net's residual (prediction − label) mean and variance in each cell. **Bar: kill unless
  at least two of buckets 3/4/5 show a queen-vs-no-queen residual-mean difference
  exceeding 2σ of the within-cell bootstrap.** If a single head is already unbiased in
  both sub-populations, a second head has nothing to learn.
  **Null panel:** run the same split on a *random* binary attribute of the same
  marginal frequency; if that also fires, the statistic is broken.
- **Cheap upside variant if the kill test passes weakly:** keep 8 buckets but move the
  boundaries. Our `OUTPUT_BUCKETS_LAYOUT` puts 9 of 33 piece-counts in bucket 0 and the
  measured effective bucket usage on our real distributions is **5.67–7.36 of 8** — we
  are not even using the buckets we have.

### R4 — Make the threat channel incremental (kill `at_full`'s replay)

- **What:** perf study 8's own #1 next item. `at_full`'s replay plumbing is **5.73% of
  all engine instructions and SF has no equivalent stage at all** [M]; unifying our two
  independent threat systems (Wave-G ordering map + NNUE FullThreats event stream,
  **397 i/n together against SF18's single `update_piece_threats` at 304 for the same
  two jobs**) is worth another ~1.3%.
- **Why here:** it buys speed with **zero eval risk** — the change is node-identical and
  eval bit-exact by construction. In a study where every other speed lever trades eval
  quality, that is a different risk class.
- **Expected Elo:** ~7% of instructions ⇒ **+12 at 120, +7 at 70** [D].
  **P(survives): 0.75** — the mechanism is known and the measurement instrument is
  deterministic; the risk is engineering difficulty, not outcome.
- **Cost:** 5–10 engineer-days, 0 GPU. Perf study 8 calls the unification "a design
  exercise, not a patch".
- **Pre-registered kill test:** the study's own `arm8.sh` gate — eval hash unchanged,
  d13 node total unchanged, `instructions:u` on a fixed 5 M-node tree. **Bar: kill any
  arm that does not deliver ≥3% instructions; six of study 8's twelve arms failed
  exactly this way** (−0.045%, +1.31%, +0.19%, +2.2%, +1.04%, nothing).
- **Price first, design second:** study 8's `probe8.sh` stub-and-diff should put a
  number on the candidate before anyone writes the correct version. It would have saved
  two of that study's six failures.

### R5 — The sibling-ranking loss term (NOVEL; see §8)

- **Expected Elo:** +10 to +25 [E]. **P(survives): 0.30.**
- **Cost:** ~5 engineer-days (trainer + data extraction) + 50–60 GPU-hours.
- **Pre-registered kill test:** §8.3. **Zero training cost, ~10 minutes of one core.**

### R6 — Deploy the uncertainty / WDL head — but run the head-free control FIRST

- **What:** train `l3_var_weights` under Gaussian NLL alongside the value mean (the
  loader path already exists and is `None`-safe), and consume σ̂ in the aspiration
  window width and the RFP margin.
- **Why:** the measured defect is under-resolution in balanced positions; a per-position
  confidence is the direct instrument for spending search where the eval is unsure. Per
  `design/17` §3.6 it "cannot corrupt correctness, only spend".
- **The economics are better than they look**: the consumer side is **already built and
  currently inert** (§4.10) — `ARCHON_UNC_TM_C2` and `ARCHON_WDL_TM_C3` in
  `src/search/mod.rs`, verified byte-identical-to-baseline on D48X because the net
  carries neither head. **Zero engineer-days on the consumer.**
- **And the motivation is measured**: we spend **18% less of our time budget than SF18
  in balanced positions and 33% less when clearly winning** [M] — and "clearly winning"
  is where we convert 12% of won books to SF's 84%.
- **Expected Elo:** +3 to +10 [E]. **P(survives): 0.30.**
- **Cost:** ~1 engineer-day (the consumer exists) + ~20 GPU-hours for the head, **but
  the first two experiments cost 0 GPU-hours.**
- **Pre-registered order — the two free experiments come first.**
  1. **Establish the SIGN, head-free.** `ARCHON_WDL_TM_C3` is written as
     `1 − c3·decisiveness`, i.e. it *shortens* the search when the position is decisive
     — the opposite of what the 33%-under-allocation-when-winning measurement suggests.
     Drive both factors from the **search's own root score** (a head-free proxy that
     needs no net change) and test both signs. **If the head-free version is null in both
     directions, the head is not worth training.**
  2. **Beat the head-free substitute, not nothing.** The sidecar's bar is the head-free
     arm's result, not zero. This is the control that stops a positive result being
     mis-attributed to the head.
  3. Only then: train the head, and before any search consumption, check that σ̂ is
     *calibrated* — bin heldout positions by σ̂ and verify the realised residual std
     increases monotonically across bins with a Spearman ≥ +0.8. An uncalibrated σ̂ is
     worse than a constant. Note the review finding already in the code: σ̂ is a
     **WDL-space** std, not a cp-space one (`sota_head.rs:165-168`), so any cp-space
     consumer must convert.

### R7 — The isolated threats-off measurement, at matched width

- **What:** train one HalfKA-only net at **1280** (not 2048), everything else identical,
  and measure it against D48X at fixed nodes and at equal time.
- **Why at 1280 and not 2048:** the staged `hka2048_probe.json` bundles *three* changes
  (threats off, width 1280→2048, neck 16→32) and therefore cannot isolate anything.
  Threats have never been isolated in this project's history; exp-0065's +32.5 ± 17.3
  bundles the whole D40 arch. **Measure the thing, then choose the width.**
- **DEMOTED by the field survey.** [S, §4.2/§5.4] Nine of thirteen surveyed top-20
  engines carry threat features, including all four rated above us, with measured gains
  of **+26 to +62 Elo** (Monty +48.06 ± 3.3, Stormphrax 8.0.0 +62.4 STC / +81.9 LTC,
  PlentyChess +45.49 fn, Pawnocchio +26.71 STC / +30.32 LTC, SF18 "up to 46"). Every one
  shows the same signature D48X shows — positive at fixed nodes, flat-to-negative at
  STC, positive at LTC. **The most likely reading of "our threats look expensive and
  unproven" is that we are testing them at the time control where nobody's threat result
  is positive** (see R0b). I would not spend a GPU-day here before R0b is answered.
- **Expected Elo:** the arm itself is +15 to +47 at 120 / −5 to +28 at 70 (§4.2), but
  the field's evidence pushes the eval-cost term toward the top of its range.
- **P(survives): 0.15** (cut from 0.35 on the external evidence). **Cost:** ~2
  engineer-days (the 10-line Pallas fix is known, `strip_threats.py` gives the warm start
  free) + 50–60 GPU-hours.
- **Pre-registered kill test, if run at all:** fn50k at sb30 and sb60 against the frozen
  D48X yardstick, **plus an LTC arm**. **Bar, set before launch:** kill if sb60 is worse
  than −40 fn Elo. Note the D48X precedent: five instrument rounds ran to sb150 at a flat
  −57 before the run was stopped.
- **The reason to run it anyway is not Elo, it is that threats have never been isolated
  in this project's history** and two sessions of argument have been built on a refuted
  number. If it is run, run it **at matched width (1280)**, not bundled with the
  width-and-neck changes the staged `hka2048_probe.json` carries.

### R8 — Prune high-churn threat features

- **What:** measure per threat-feature-class (a) update rate per move and (b) trained
  weight mass, and drop the classes with high churn and low mass. SFNNv11 did exactly
  this — it dropped piece→king threats, "saving 13 MB of net space and approximately
  **0.8 threat feature updates per incremental accumulator update**", non-regression
  [S, §5.4].
- **Why it fits us specifically:** our threat cost is dominated by *event* work —
  `at_full` replay 5.73% + event generation 3.55% + fold 10.64% — all of which scales
  with the number of active and changed features, not with the table size. **Dropping
  features is the one threat-side change that reduces both the table and the churn
  without changing the encoding.**
- **Expected Elo:** +2 to +8 [E]. **P(survives): 0.45.**
- **Cost:** ~2 engineer-days + a retrain (or, cheaper, fold it into whichever run happens
  next). Note we already have SFNNv11's specific prune: our `MAP` excludes the king as
  both attacker and victim (`sota_features.rs`).
- **Pre-registered kill test — offline and nearly free:** extend
  `threat_collapse.py`'s index inversion to bucket the trained rows by class and report
  mean row norm per class; separately instrument one bench run for events-per-move by
  class. **Drop-candidate = a class in the bottom quartile of mass and the top quartile
  of churn.** If no class satisfies both, there is nothing to prune and the item dies for
  the cost of one afternoon.

### R9 — Reject / defer

Dual big/small nets (§4.7, SF removed it; ceiling +7 to +14 for weeks of work), width
changes in either direction (§4.1), neck widening (§4.3), policy (§4.9), a third hidden
layer beyond R2's (no precedent in any of the three references), and 3072-wide anything
(RAM). Full reasoning with numbers in §9.

### Summary table

| # | item | E[Elo]@120 | E[Elo]@70 | P | eng-days | GPU-h | kill test cost | evidence class |
|---|---|---:|---:|---:|---:|---:|---|---|
| R0 | measure Elo/doubling | gate | gate | 1.00 | 0 | 0 | — | — |
| R0b | add an LTC arm to every arch gate | gate | gate | 1.00 | 0 | 0 | — | external [S] |
| **R2** | **dual activation (+ concat skip, gated separately)** | **+15** | **+17** | **0.60** | 4 | 1 (+60) | **1 GPU-h** | **external A/B [S]** |
| R1 | complexity channel | +6 | +6 | 0.35 | **0.5** | **0** | 600 games | SF source [M] |
| R4 | incremental threats | +12 | +7 | 0.75 | 8 | 0 | 1 eng-day | our profile [M] |
| R3 | head-bucket decoupling + queen split | +10 | +10 | 0.40 | 2 | 60 | **0** | our data [M] |
| R6 | uncertainty / WDL head (head-free control first) | +6 | +6 | 0.30 | **1** | 0 then 20 | **0** | consumer exists [M] |
| R8 | prune high-churn threat features | +5 | +4 | 0.45 | 2 | 0–60 | 1 afternoon | external [S] |
| R5 | sibling-ranking loss (novel) | +18 | +18 | 0.30 | 5 | 60 | **10 core-min** | our defect shape [M] |
| R7 | threats-off @1280, isolated | +30 | +12 | **0.15** | 2 | 60 | 2×5 GPU-h | field says no [S] |

(Elo columns for R1/R3/R5/R6/R8 barely move with the speed constant because those items
are eval-quality or near-zero-cost changes; R2's *improves* at 70 because its speed cost
shrinks; R4 and R7 are pure speed and shrink with it.)

**If only one thing is done: R2.** It is the only item with an external, standalone,
controlled A/B behind it (+6.20 STC / +12.05 LTC), it costs 2.3% of instructions, it
needs no width or feature-set change, and it has a 1-GPU-hour kill test.
**If only one *hour* is available: R1** — 0.5 engineer-days, no retraining, and it
consumes an output the architecture already computes and throws away.

---

## 8. THE NOVEL PROPOSAL — rank-aware value training

### 8.1 The structural argument (why ARCHON's optimum differs from SF's)

Three measured facts, held together:

1. **99.84% of our parameters are in the feature transformer and 0.16% are in the
   head** [M, §1.2], and **exactly one bucket's 22,353 head parameters are touched per
   eval** [D].
2. **Two-thirds of our eval instruction cost is accumulator maintenance, one-third is
   the head forward** [D, §1.3]. Every capacity axis except one costs per-node time:
   width scales maintenance *and* the head; neck and depth scale the head.
3. **Our binding constraint is per-node cost and SF's is not** — we are 1.4× slower in
   games [M] and 1.512× on instructions after perf study 8's changes [M].

⇒ **The only capacity axis that is free at inference is the one that does not add
compute at all: better *targets* and better *routing*, not more neurons.** SF, Reckless
and Integral all sit at 8 output buckets and pointwise value regression because their
per-node budgets let them buy capacity the ordinary way. Ours does not. **We should
expect our optimum to be further along the "free capacity" axis than theirs, and that
is a structural argument, not an analogy.**

R3 (routing) is the *routing* half of this. R5 is the *targets* half, and it is the
genuinely new idea.

### 8.2 The proposal

**Train the value head with an added pairwise ranking term over sibling positions.**

Today's objective is pointwise: minimise `(v(s) − target(s))²`-ish over independent
positions. A pointwise loss is **indifferent to the ordering of positions whose targets
are close** — it is optimised by being unbiased, not by being discriminative. But a
search does not consume `v(s)`; it consumes `argmax` over siblings. What decides a game
is whether `v(child_a) > v(child_b)` matches the oracle, and that is a *ranking*
property that the current loss never asks for.

**This is exactly the shape of our measured defect.** Our eval is unbiased (MAE 97.1 vs
SF18's 97.2, p90 *lower*) and we still lose 68 Elo at equal nodes and equal depth, with
the loss concentrated in near-equal middlegame positions where we churn 1.561× SF and
convert 12% of won books against 84%. **Calibrated but under-discriminative** is
precisely what a pointwise loss produces and a ranking loss fixes.

**The concrete term.** For a parent `p` with children `a, b` and an oracle preference:

```
L_rank = Σ_{(p,a,b)}  w(p,a,b) · softplus( −(v(a) − v(b)) · sign(oracle(a) − oracle(b)) / τ )
w(p,a,b) = 1 if |oracle(a) − oracle(b)| is SMALL   (the band we lose in)
```

with the weight deliberately **inverted** relative to intuition: the pairs worth
training on are the *close* ones, because the easy pairs are already ordered correctly
and contribute no gradient we need.

**Why ARCHON can do this and the reference engines cannot, cheaply.** We already have
the data infrastructure: the Observation Store (`design/16` §3.5, D29/D43) records
Root Move Analysis and Counterfactual pairs — literally `(parent, move_a, move_b,
oracle preference)` tuples — and `tools/nnue/root_move_oracle.py` already produces them.
Bullet and nnue-pytorch have no such record; their training data is a flat stream of
independent positions. **This is a lever our data platform was built for and that our
trainer has never used.**

**Cost at inference: exactly zero.** It is a loss term. Every node in every game
forever pays nothing for it.

### 8.3 The pre-registered kill test — 10 core-minutes, no training

**Claim to falsify:** "ARCHON's value function ranks close siblings worse than SF18's,
at equal pointwise accuracy."

**Procedure.**
1. Take the 308 `searchopt/positions/plateau-advside.json` positions (these are exactly
   the unconverted-win plateau positions where the drift lives) and enumerate legal
   moves.
2. For each parent, static-evaluate every child with **our** net and with **SF18's**,
   via `searchopt/tools/staticeval.py` — which already handles the two POV conventions
   and the 2.07× frame at the tool boundary, the trap that has been hit twice in this
   repo.
3. Score every child with a deep oracle (SF18 at 120k nodes, `searchmoves` per child) —
   or, cheaper, use the existing `cp_white`/`depth` labels already in the file for the
   parents and generate oracle child scores only for the top 8 moves.
4. Statistic: **pairwise ranking agreement with the oracle among child pairs whose
   oracle scores differ by < 50 cp**, ours vs SF18's.

**Bars, set now.** If our close-pair agreement is within 2 percentage points of SF18's,
**the hypothesis is dead** and R5 is dropped — our ranking is as good as our pointwise
accuracy suggests and the deficit is elsewhere. If we are ≥5 points worse on close
pairs while matching on far pairs, the hypothesis is confirmed with a mechanism.

**Null panel, run first (rule 1).** Compute the same statistic with our net compared
against *itself* under an inert ±1 cp dither. If an inert perturbation moves close-pair
agreement by more than 2 points, **the instrument cannot answer this question** and the
test must be redesigned before it is read. The 44-position forced-move instrument in
this repo failed exactly this check and was read twice, in opposite directions.

**Threat to validity, stated up front:** static eval of a child is not what the search
consumes (it consumes a searched value), so a null result bounds the static-ranking
hypothesis only. The follow-up, if the static test is null but the suspicion survives,
is the same statistic on *searched* child values at fixed nodes.

### 8.4 A second, smaller novel item, for free

**Regularise the threat transformer's weight norm during training so it fits the 255
frame.** [D, §4.8] The reason our two transformers live in different frames is *range*:
the trained net reaches |w| = 126 where only |w| ≤ 31 survives a rebase, so 0.031% of
weights saturate and the eval tail reaches 205 SF cp. A weight-norm penalty on the
threat FT during training could land the trained weights inside the 255 frame, at which
point the merge becomes a bare `vec_add_16` — SF's shape — and **~2.5% of engine
instructions come back permanently** (perf study 8 §6.1 measured the `uniquant` arm at
−2.71%, static accounting 2.47%).

This has never been tried, costs one hyperparameter in the trainer, and is a rare case
where a *training* constraint buys *inference* speed. Attach it to whichever full run
happens next; the kill test is free (measure `max |threat_w|` at each checkpoint and
compare to 31).

### 8.5 A third idea, and the measurement that killed it

**Recorded because it is the most interesting thing I did not recommend.** Vine — the
MCTS engine by the same authors whose data trains Pawnocchio — encodes threats not as a
59,808-dim relational space but as **two extra bits on the piece-square feature**:
`ft_weights[is_defended][is_threatened][is_enemy][piece_type][square]`, **3,072 inputs
in total** [S].

**Why it looked exceptionally well matched to ARCHON.** Our threat channel costs ~24% of
all engine instructions, and essentially all of that cost exists *because* the feature is
relational: 77.7 MB of table, up to 128 active features per perspective, up to 80 events
per move, a separate accumulator stack, and the `at_full` replay stage SF does not have.
A flag encoding collapses every one of those — and, uniquely for us, **the engine already
computes the input**: the Wave-G board map maintained for move ordering is exactly
"every square attacked by the side-not-to-move, split by attacker type"
(perf study 8 §4.3). It would have unified the two threat systems (perf study 8 item 8)
and deleted the replay stage in one design. Estimated saving ~16–18% of instructions ⇒
**+30 to +35 Elo at 120, +18 to +20 at 70** [E].

**The measurement.** `docs/research/nnue-arch-artifacts/threat_collapse.py` re-derives
the exact index construction from `sota_features.rs` (empty-board geometry → prefix-sum
offsets → dense to-ordinal), enumerates every `(attacker, from, victim, to)` tuple
forward to invert index → relation (89.1% of the 60,720 rows resolve; the residual is my
enumeration's handling of the semi-excluded direction, not the net's), and runs an ANOVA
over the trained 1280-dim rows: `R² = 1 − SS_within/SS_total`. **Null panel run first
(rule 1): the same statistic with group labels shuffled — same cardinality, same group
sizes, no structure.**

| grouping of the trained threat rows | groups | R² | null | **R² − null** |
|---|---:|---:|---:|---:|
| **(victim type, victim square, is-enemy-attacker) — the flag encoding** | 640 | 5.3% | 1.2% | **4.1%** |
| (victim type, victim square, victim colour) | 640 | 4.2% | 1.2% | 3.0% |
| (attacker type, victim type, victim square) | 1,326 | 6.4% | 2.4% | 4.0% |
| (attacker type, victim type) | 21 | 1.4% | 0.0% | 1.3% |
| (attacker type, attacker square) | 304 | 1.9% | 0.6% | 1.3% |
| (victim square only) | 64 | 0.7% | 0.1% | 0.6% |

Restricting to the top half of rows by L2 norm — the features that actually carry mass —
changes nothing (4.8% for the flag grouping).

**Verdict: killed.** No coarse grouping of the relation explains more than ~5% of the
trained table's row variance above its own null; even a 1,326-group partition manages
4.0%. **The learned content of our threat channel lives in the specific
(attacker, from, victim, to) relation, not in a per-piece attacked/defended flag.** A
flag encoding would have to relearn the function rather than inherit it, and there is no
strength anchor for it in any A/B engine (Vine is MCTS and unrated on CCRL). Against
that, nine of thirteen top-20 A/B engines ship the *relational* form with measured
+26 to +62 Elo (§5.4).

**Caveat kept honest, because it is the same caveat as the "zero the threat rows"
number:** this is a weight-space statistic on a net *trained* with the relational
encoding. A net trained with flags would put the information elsewhere. What the
measurement establishes is that the flag scheme is **not a cheap re-encoding of what we
already learned** — it is a different net and a full retrain with no external evidence
behind it. That is enough to keep it off the ranked list.

---

## 9. WHAT I WOULD NOT DO, WITH NUMBERS

| option | why not | the number |
|---|---|---|
| **Widen to 2048 or 3072** | speed cost exceeds the eval gain at the project's own constant, and 3072 is a 331 MB net on a 15 GB box where RAM is already the binding resource | −41.5 / −84.9 Elo of speed at 120 [D]; eval gain +28 / +55 at +44/dbl [E] |
| **Narrow to 512** | 92% of the owner's proposal, and it is the trade already measured as a wash | +54.8 speed vs −58 eval [D] |
| **Widen the neck (16 → 32 or 128)** | the neck→L3 map spans 6–11 effective directions at *every* width tested; 128 runs at 1/16 utilisation against its null | obs/null 0.841 @16, 0.603 @32, **0.238 @128** [M, new] |
| **Narrow the neck to 8** | +2.5 Elo at 120, and half of that is weight traffic only — a zmm holds 16 i32 lanes so `l1_dot` still issues one `vpdpbusd` per group | +2.5 Elo [D] |
| **Dual big/small nets** | our own constants reference **deleted them**, and SF's removal commit says both non-regression SPRTs *failed* and it merged anyway because "the small net's lower eval quality is deleterious"; our honest ceiling is +4–8% NPS for weeks of work and a TT-consistency risk | +7 to +14 Elo at 120 [M, perf study 7 §4/§7]; `ab8f901d` [S] |
| **Threat-as-flag re-encoding (Vine-style)** | the most attractive idea I found and I killed it with my own measurement: no coarse grouping of the relation explains more than ~5% of the trained threat table's row variance above its null | flag grouping **R² − null = 4.1%**; even 1,326 groups = 4.0% [M, new, §8.5] |
| **x-ray threats / psq-suppression** | measured elsewhere, badly | **">50 Elo fixed-nodes loss in Monty"** [S] |
| **A third hidden layer** | no precedent in SF, Reckless or Integral; our tail is already 2.16% of instructions for 544 MACs and R2's version is the evidence-backed increment | R2 costs +2.29%; a third layer costs more for no cited precedent |
| **Return to policy** | co-training measured **−14.0 ± 23.0** on the eval; a deployed 17% top-1 policy measured **exactly zero** Elo over 2×300 games; the bar is 35% and the best ever seen is 27.2% | [M, exp-0121 and the sb30 ablation] |
| **Rebase the threat weights into the 255 frame at load** | 0.031% of weights saturate at i8 and they are the *largest* ones | mean error 3.97 SF cp, **max 205 SF cp** [M, perf study 8 §4.5] |
| **Store threat rows pre-widened at i16** | bit-exact and −2.10% instructions, and it still loses: the table goes 78 MB → 155 MB against a 32 MB L3 | cache-misses ×1.299, cycles ×1.078 [M, perf study 8 §7.11] |
| **Sparse `skip_dot` over the nnz list** | FT group density is 36.5%, not the 5–15% the source comment claims, so 16-element block occupancy is 84% | ceiling 0.4% [M, perf study 8 §3.2] |
| **King-bucket factorization (Integral's)** | tested here and null — but note the test was **underpowered** at ±18 for a plausibly +10 effect; if a run is happening anyway, re-arm it, do not re-litigate it standalone | +6.9 ± 18.0, n=300 [M, exp-0099] |
| **Any equal-time match to settle an architecture question at n ≤ 400** | `searchopt/LEDGER.md`: 406 games gives ~40–50 Elo observed half-width. Every one of exp-0149…0155's arch verdicts sits inside its own bar | n=406 ⇒ ±40–50; n=1,000 ⇒ ±15 [M] |

---

## 10. IS THE CURRENT ARCHITECTURE OPTIMAL? — the honest answer

**Close to it, on the axes that have been searched; unsearched on two axes that are
cheap.**

What the evidence supports:
- The **width** is right and is fully paid for (1.080× justified premium, two
  independent derivations agreeing to 0.3%) [M].
- The **feature set** is a faithful transcription of the strongest engine's, at the
  *identical* 60,720 threat dimensions as our constants reference [M, verified].
- The **neck** is at or slightly above its knee, and the standing suspicion that it was
  the bottleneck is retired by measurement [M, new].
- The **quantisation** shape (int8-sparse FT/L1 → f32 tail, QA=255, `FT_SHIFT=9`) is
  independently converged on by three top engines and its one asymmetry is a deliberate
  footprint decision, not a mistake [M].
- The **skip connections** at the feature end are already stronger than SF's [M].
- The **known deficiencies are implementation, not architecture**: 5.7% replay, 1.3%
  duplicated threat bookkeeping, 6.1% unprofiled make/unmake, 2.0% accumulator memcpy
  [M, perf study 8].

What has never been searched:
- The **tail's shape** (dual activation + inter-layer concat), which is where SF moved
  after we copied it, which the rank measurement independently points at, and which
  eleven of thirteen surveyed top-20 engines have and we do not. **This is the one place
  the architecture is genuinely behind the field**, and it is also the cheapest to fix.
- The **output-bucket axis**, which is the only free-at-inference capacity we have and
  which every reference engine leaves at 8 — from a per-node budget position we do not
  share.
- The **loss's indifference to sibling ordering**, which matches the measured defect
  more exactly than any capacity story does.
- **Pawn-pair features** — 4,560 rows, adopted by Stockfish within five weeks of
  Pawnocchio inventing them, and not priced anywhere in this study (§12).

And one thing that turns out not to be an architecture question at all: **the field's
single biggest reported win of the year (+28 to +46 Elo from the eval-scale constant) is
a bug-class D46 closed for us before it could cost anything** (§4.9), and our own two
tests of it came back negative in both directions.

**A well-evidenced null is worth more than a speculative redesign, and most of this
study is that null.** The items I would actually spend on — R2, R1, R4, R3 — total
14.5 engineer-days and ~121 GPU-hours, and all four have kill tests costing one GPU-hour,
600 games, one engineer-day and **zero** respectively. Two of the study's own ideas were
killed by its own measurements before reaching that list (§8.5, §4.3).

---

## 11. WHERE ESTIMATES HAD ALREADY HARDENED INTO FACTS

Recorded because the repo's stated failure mode is exactly this, and I found three
live instances while writing.

1. **"The threat channel costs 1.78× / half our speed" → real figure 1.25–1.39×.**
   Refuted by perf study 7 §5 on 2026-07-27, but `Cargo.toml`'s `arch-hka2048` comment
   still quotes 1.78× *and* carries the correction in the same paragraph, and
   `docs/research/value-foundation-state.md`'s "D49 ~+50 Elo" recommendation — still the
   #2 item in that document's runbook — is sized on the refuted numbers. **§4.2 re-derives
   it: the bet survives but its margin halved.**
2. **"Width becomes 5.7× cheaper without the threat channel" → 1.81×** [D]. Same
   session, same refutation, but the derived multiplier was never re-derived; the whole
   "optimum ~2048–2560" width table in `value-foundation-state.md` rests on it.
3. **"SF18's threat feature space is LARGER than ours"** (`searchopt/LEDGER.md`) — true
   at the `sf_18` tag (79,856), **false at our own constants reference** `eca43a97`
   (60,720, exactly ours). This is the same class of error as the razoring-465 case
   CLAUDE.md documents: a shape question answered against the checkout rather than the
   reference.

And one repo-hygiene finding: **`integral_reference/` was deleted from the working tree
in commit `1f8c111`**, apparently as collateral to the D48X promotion commit. Its 15
tracked notes are recoverable with `git show 1f8c111^:integral_reference/<file>`.
`reckless_reference/Reckless/` is also absent (gitignored clone, never present here),
so §5.2 could not be verified against source.

---

## 12. SOURCE PROVENANCE AND WHAT COULD NOT BE VERIFIED HERE

**Pawnocchio: found, real, and covered in §5.3** —
`github.com/JonathanHallstrom/pawnocchio`, Zig, GPLv3, CCRL 40/15 4-CPU **3643, #4**
(list of 2026-07-23). Its architecture was read from source on GitHub and its 59,808
threat dimension and 68.4 MB net size were re-derived arithmetically and matched.

**What is [M] — I ran it or read it in this repo:** everything in §1, §2, §3, §4.1's
ARCHON rows, §4.3, §4.5, §4.6, §4.8, §4.9's ARCHON half, §4.10's ARCHON half, §6, §8.3–8.5,
§9's ARCHON rows, §11, plus the Stockfish rows of §5.1 which I verified directly at
`eca43a97` and `sf_18` in the local clone.

**What is [S] — read from the public web by a subagent and NOT verifiable in this
repo:** all of §5.2 (Reckless/Integral — see below), all of §5.3 and §5.4, the
Stormphrax/PlentyChess/Monty/Yukari measured deltas, the cosmobobak eval-scale result,
and the SFNNv16 claims below. These are used to *raise or lower confidence* in items
whose primary evidence is local; **no recommendation rests on an [S] source alone**,
with one deliberate exception: R2's promotion from "estimate" to "best-evidenced" rests
on Stormphrax PR #239, and it is flagged as such at the point of use.

**Three verification gaps the owner should know about.**

1. **`integral_reference/` was deleted from the working tree** in commit `1f8c111`
   (15 files, 8,122 deletions, alongside `GEMINI.md`) — apparently collateral to the
   D48X promotion commit. Recover with
   `git show 1f8c111^:integral_reference/<file>`. **`reckless_reference/Reckless/` is
   also absent** (gitignored clone, never present in this working tree), so §5.2 could
   not be checked against source at all.
2. **The Stockfish clone is 12 days stale.** `origin/master` in the local clone is at
   2026-07-16 (20 commits past `eca43a97`), still `L1 1024 / L2 32 / L3 32`,
   `FullThreats::Dimensions = 60720`, net `nn-0ee0657fb25e.nnue`. The survey reports
   that **SFNNv16 landed 2026-07-20 and adds a 4,560-dim pawn-pair feature set**
   (`f4bcd404`, co-authored by Pawnocchio's author), taking the input space to
   22,528 + 59,808 + 4,560 = **86,896**, with the net now `nn-ab28990d4ea3.nnue`
   (2026-07-26, 95,144,073 bytes by sha256). **I could not verify any of that locally.**
   A `git fetch` would settle it, and it is worth doing before the next transcription
   pass: it would be the first *feature-set* change since we transcribed, and the
   arithmetic 22,528 + 59,808 + 4,560 = 86,896 does check out.
3. **Pawn-pair features are the one genuinely new input family in the field and this
   study does not evaluate them.** They are cheap (4,560 rows against our 60,720),
   they were measured at +26.71 STC / +30.32 LTC in Pawnocchio bundled with threats,
   and SF adopted them within five weeks. **That is the highest-value thing this study
   did not have time to price**, and it should be the first question of the next one.

---

## 13. ARTIFACTS AND REPRODUCTION

All under `docs/research/nnue-arch-artifacts/`. Pure Python; `net_anatomy.py`,
`neck_rank.py` and `neck_truncate.py` need numpy (use `.venv/bin/python`).
None of them starts an engine or a search.

```bash
# parameter accounting, FT channel mass, neck spectrum, PSQ1/skip stats
.venv/bin/python docs/research/nnue-arch-artifacts/net_anatomy.py \
    engines/canonical/net.cenn

# neck capacity WITH ITS NULL PANEL (four nets, three neck widths)
.venv/bin/python docs/research/nnue-arch-artifacts/neck_rank.py \
    data/nnue/nets/screens/d48x-1280-sb520-ema.cenn \
    data/nnue/nets/screens/d47v-a4b2-cal400.cenn \
    data/nnue/nets/screens/d47s-v15-sb230-ema-rawframe-nocpol.cenn \
    data/nnue/nets/d40-uho-v1-240sb.cenn

# how much of the neck->L3 map survives truncation to k dims
.venv/bin/python docs/research/nnue-arch-artifacts/neck_truncate.py <nets...>

# output-bucket occupancy + P(queen | bucket) on real ARCHON distributions
python3 docs/research/nnue-arch-artifacts/bucket_occupancy.py \
    searchopt/positions/plateau-advside.json \
    searchopt/positions/evalbench.json \
    data/losscorpus/lossmine-0727.fens \
    tools/nnue/calib/scale-check-800.fens

# the cost model, its validation against exp-0156, and the option table
python3 docs/research/nnue-arch-artifacts/cost_model.py
```

**Primary sources leaned on:** perf study 8
(`git show perf-study-8-nps-parity:docs/research/perf-study-8-nps-parity-2026-07-28.md`),
perf study 7 (`old/docs/research/perf-study-7-sf-side-by-side-2026-07-27.md`),
`searchopt/LEDGER.md`, `ledger/experiments.jsonl` (exp-0022, 0052, 0058, 0065, 0099,
0120, 0121, 0122, 0149–0163), `design/11` §12, `design/17` §3.5–3.7,
`engines/canonical/MANIFEST.md`, and — read directly, not from memory —
`stockfish_reference/Stockfish` at `eca43a97` and `sf_18`.
