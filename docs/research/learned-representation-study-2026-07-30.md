# The learned representation — Tier-0 update-locality census + the trained-properties panel (2026-07-30)

**Status:** Tier-0 measured and landed (counters, censuses, budget line, one
major attribution correction of a predecessor's price, one learned artifact
that passes its value screens); Tier-1 built, pre-registered, smoke-verified
on CPU, **deliberately not launched** (the GPU is committed to D49; the
launcher `tools/nnue/updloc_panel/updloc_t1_launch.sh` is written and
unarmed). Branch `learned-repr-study`. Predecessors:
`learned-feature-code-study-2026-07-30.md` (design),
`recode-poc-2026-07-31.md` (compression POC),
`richer-representation-study-2026-07-31.md` (refinement pricing — one of
whose central prices this study corrects by ~7×, §2).

**Verdict in one paragraph.** The incumbent representation's own
update-locality — the reference point every learned representation must
beat, never previously measured — is now pinned on the pinned frame with
exact counters (§1): a made move records 2.22 HalfKA placement edits and
3.22 raw threat events; the engine actually folds 2.23 + 3.92 rows/node
after deferral (fold-time survival 0.61); an evaluated position holds ~12.9
active HalfKA and 10.7 active threat rows per perspective; and 29.3% of all
in-tree moves relocate a king. Building the budget line exposed that **the
richer-representation study priced every context bit off the wrong symbol**:
`at_full` is the per-eval fold-orchestration entry point (0.431 calls/node,
measured), not the from-scratch rebuild (0.0196/node, whose real symbols
cost 0.71% of cycles, not 5.50%) — so a threat rebuild costs **~1,635
cycles, not ~12,700**, and every price in that study's §1.3 table is **2–8×
too high** (§2). At corrected prices, king-conditioning the threat channel
is nearly free — and a **learned** min-flip partition of the measured
in-tree king-transition matrix (§3) finds a bit (`pocket` = own king in
{g1,h1,h2}) that carries 82% of rank1's conditioning information (MI 39×
the null control) at 60% of rank1's flip price, i.e. **−1.2 Elo** against
the predecessor's −12.4-Elo-class prices. Successor pairs for a trained
update-locality objective turn out to be **free at the loader** (§5): every
binpack-derived record already carries the played move, and the D46 filter
guarantees it is a pure relocation — validated end-to-end on 200k pairs.
Tier-1 (§6) makes sparsity and locality *trained* properties: four
pre-registered arms with a content-free control and fixed decision rules.

---

## 0. Frames and denominators (stated first, because a 2× error lived here)

**The pinned frame** everywhere below:
`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
(50 FENs × 100k nodes, 1 thread, 16 MB TT, canonical D48X net
`d48x-1280-sb520-ema-s600.cenn`). Node totals per binary: canonical PGO
`archon-s25` 5,001,566; this study's instrument and clean builds 5,001,562;
b40 5,001,562 — the ±4 is node-limit stop granularity (the d13 depth frame
is **bit-identical** across all of them at the gate pin 1,739,588, so the
trees are the same; per-node normalisation absorbs the tail).

**Cycles/node, re-derived** (7-round interleaved battery, canonical PGO
binary, `perf stat` cycles:u/instructions:u, startup subtracted via nodes=1
runs; trainer live, load ~6/16; artifact `updloc-artifacts/denom_battery.tsv`):

| statistic | value |
|---|---:|
| cycles/node, median (startup-subtracted) | **4,513.9** |
| cycles/node, min round (least contention) | 4,393.7 |
| instructions/node (deterministic across rounds to 1e-7) | **5,504.9** |
| in-search IPC | 1.22 |
| startup (net load etc.), median | 778.5 M cycles / 781.2 M instr |
| quiet estimate (study 40 implied 4,104; coordinator's raw 4,489.8 loaded) | **~4,100** |

This **confirms the coordinator's resolution of the 2× contradiction**:
study 40 is right, study 36's per-node totals line (8,262.9) is wrong by
~1.84× and is never used here. Study 36's within-profile symbol *shares*
are self-normalising and are used, cross-checked against this study's own
b40 profile (§2). Elo pricing throughout: ~120 Elo/speed-doubling,
`ΔElo = 120·log2(1 + Δc/C)`; heldout↔Elo exchange 12–28e-6/Elo (inherited,
§8).

## 1. Tier-0 — the incumbent's update-locality, measured

Instrument: four new counter families behind the `foldcount` feature
(`src/eval/nnue/foldcount.rs`; commit `962fb798`) — unconditional MAKE-time
deltas (every prior counter tallied at fold time, i.e. after deferral),
an instrument-only eager `build_deltas` replay for make-time threat events,
an active-rows census at every full eval, and the 64×64 color-relative
king-square transition matrix `KTRANS`. **Neutrality gate:** the
instrumented build reproduces the wl50 eval-hash pin `2e1ede95…79c4`
exactly and the d13 node pin 1,739,588 exactly.

### 1.1 Per made move (unconditional; make.calls ≡ node total, verified)

| quantity | wl50×100k | d13 frame |
|---|---:|---:|
| HalfKA placement edits/move (DirtyPiece adds+subs) | 2.223 | 2.229 |
| — quiet (1a1s) / capture+ep (1a2s) / castle | 77.7% / 22.3% / ~0¹ | — |
| raw threat events/move (eager observer) | **3.225** (p50 2, p90 9, p99 16) | 3.606 |
| king moves/move | **0.293** | 0.280 |
| mirror crossings/move | 0.0339 | 0.0335² |

¹ zero castling on this suite — wl50 positions carry no castling rights; a
game-phase-balanced suite would show a small 2a2s cell.
² matches the richer study's d13 measurement exactly — instrument
cross-validated against theirs.

### 1.2 Per node actually paid (fold time, after deferral)

| site | calls/node | rows/node | rows/call |
|---|---:|---:|---:|
| HalfKA fold (`hka_into`) | 0.978 | 2.230 | 2.28 |
| Finny warm refresh (`hka_dual`) | 0.165 | 0.942 | 5.72 |
| threat fold (`thr_into`) | 0.854 | **3.922** | 4.59 |
| threat empty-copy / parent-alias | 0.022 / 0.158 | — | — |
| threat from-scratch (`thr_refresh`) | 0.0196 | 0.081 | **4.16** |
| full evals | 0.431 | — | — |

Deferral interaction: HalfKA chain-folding pays ~100.3% of recorded rows
(laziness saves *evals*, not fold rows — the subtree property the threat
module doc states); threats fold 0.61 of (2 perspectives × recorded
events); refresh survival s = refreshes/crossings = 0.578 (d13: 0.566).

### 1.3 Active rows per evaluated position (per perspective)

| channel | in-tree mean (wl50) | p50/p90 | data-side (200k heldout pairs) |
|---|---:|---|---:|
| HalfKA (= piece count) | 12.93 | 11 / 24 | 14.38 |
| FullThreats | 10.70 | 6 / 30 (p99 44) | 12.31 |
| FullThreats **at refresh points** | **4.16** | — | — |

The refresh-point number is the load-bearing one: refreshes are
king-move-conditioned and therefore endgame-tilted, so a mirror rebuild
folds ~4 rows, not ~11. (Data-side actives reproduce the design study's
census — 14.42/12.36 on a different band — to 0.3%: two independent
pipelines agree.)

## 2. The attribution correction — a predecessor's price was ~7× too high

The richer-representation study priced every candidate context bit with
"`at_full` = 5.50% cycles at 0.0214 rebuilds/node ⇒ 2.57% per 0.01/node",
i.e. **~12,700 cycles per rebuild**. Three independent measurements say the
attribution is wrong:

1. **`at_full` runs at the eval rate, not the refresh rate.** This study's
   counters: `at_full` is the entry point `evaluate` calls — 0.431
   calls/node — while `refresh_threat_into_at` (where the rebuild actually
   happens) runs at 0.0196/node. At 0.431 calls/node, `at_full`'s share is
   ~585 cycles/call of fold orchestration + resolution — plausible; at
   0.0196 it would be 12,700 cycles to enumerate and fold **4.16 rows** —
   not plausible.
2. **The b40 profile on the pinned frame** (artifact
   `updloc-artifacts/perf-b40-symbols.txt`): `refresh_threat_into_at` =
   0.57%, `refresh_psqt_threat_at` = 0.14% — the actual rebuild is **0.71%
   of cycles at 0.0196/node ⇒ ~1,635 cycles/rebuild** (loaded; ~1,485
   quiet). These symbols sit *below* study 36's table cutoff, which is how
   the misattribution survived.
3. **The predecessor's own sources contradict it.** The eager-era rebuild —
   full board, no deferral — was measured at 6,240 instructions/call
   (receipt in `apply_at`'s comment, study 17). The 12,700-cycle price
   implies ~17,500 instructions per *lazy, 4-row* rebuild — 2.8× the cost
   of the eager full-board rebuild it replaced. Impossible.

**Corrected price: ~1,635 cycles per threat rebuild** (mirror-conditioned,
4.16 rows, ~13 pieces). Phase caveat: a bit whose boundary flips in
middlegames rebuilds bigger positions; the eager-era receipt bounds even a
full-board rebuild at ~6,240 instr ≈ ~5,100 cycles — so the honest range is
**1.6k measured to ≤5.1k bounded**, against 12,700 assumed. Every price in
the richer study's §1.3 table is therefore 2.5–7.7× too high; its
"input-side refinement is not free — every candidate bit costs 3.4–8.6% of
cycles (−6…−14 Elo)" verdict does not survive:

| context bit (as ADDITION, exact union increment × s × rebuild) | richer study | corrected (measured…bound) |
|---|---:|---|
| rank1 | −12.4 Elo | **−1.9 … −5.9 Elo** |
| rankhalf | −5.8 Elo | −0.8 … −2.4 Elo |
| `pocket` (learned, §3) | — | **−1.2 … −3.6 Elo** |
| learned ≥20% triangle (§3) | — | −0.6 … −1.9 Elo |

(Second correction, smaller: the richer study priced additions by each
bit's standalone crossing rate; the exact increment is
union(mirror, b) − mirror, computable from KTRANS — e.g. rank1 0.05265 vs
0.05611 standalone, a further 6% discount. Replacement pricing per the
coordinator's earlier correction is also exact from the matrix.)

## 3. Learned update-locality: min-flip king partitions from the measured matrix

`KTRANS` (color-relative from→to king transitions, both frames' counters
agree with all six hand `kmove` bits exactly) makes ANY partition's in-tree
flip rate computable without an engine run — including learned ones.
Annealed min-flip bipartitions subject to a data-side occupancy balance
(occupancy from record `ksq`/`opp_ksq` over 1M heldout positions; **frame
note and retraction**: `opp_ksq` is stored in the enemy's own frame —
verified against decoded boards — and this study's first readout
double-flipped it; every occupancy-dependent number below is post-fix, and
the pre-fix table is void):

| partition | flips/node | exact add | repl vs mirror | data min-mass |
|---|---:|---:|---:|---:|
| mirror (files a–d) — incumbent | 0.0339 | — | — | 26.5% |
| rank1 (hand) | 0.0561 | 0.0527 | +0.0222 | 38.7% |
| rank12 (hand) | 0.0549 | 0.0484 | +0.0209 | 35.6% |
| rankhalf (hand) | 0.0242 | 0.0217 | −0.0097 | **11.0%** |
| learned ≥10% (queenside corner triangle) | 0.0117 | 0.0117 | −0.0222 | 10.0% |
| **learned ≥20%** (queenside triangle) | 0.0239 | **0.0166** | −0.0100 | 20.2% |
| **learned ≥30% = `pocket` {g1,h1,h2}** | 0.0318 | 0.0318 | −0.0021 | **30.4%** |
| learned ≥40% | 0.0359 | 0.0359 | +0.0019 | 40.6% |

(Balanced on in-tree ARRIVAL mass instead of data occupancy — the census
script's variant, artifact `census_wl50.txt` — the annealer finds
0.02344 flips/node at ≥25% minority arrival mass vs the mirror's 0.03394,
31% cheaper, with advanced-wedge shapes; the mirror is matched only at
≥40% arrival balance, 0.03372. Both framings agree on the shape of the
result.)

**These are COST results, not value results** — a partition can be cheap
and condition nothing. The value screen below and Tier-1's U1/U2 carry
that burden. Findings, honestly bounded:

- **The hand mirror is well-placed for its class** — no learned partition
  at ≳26% both-sides mass beats its flip rate by a large factor. The naive
  "hand design must be far from optimal" story is false here.
- **The rank-family bits the predecessor screened are dominated**: `pocket`
  gives 79% of rank1's mass at 60% of its exact increment; the ≥20%
  triangle gives half rank1's mass at 32% of its price. Kings *shuffle
  inside* the castled pocket and *walk through* rank boundaries — a
  structure no hand rank/file cut matches.
- **rankhalf's cheapness is partly value-thinness**: its minority cell holds
  11.0% of data-side king mass (the coordinator's +2.5-Elo replacement
  trades conditioning mass the speed table cannot see).
- **Out-of-sample**: partitions learned on the wl50 matrix transfer to the
  d13 matrix within ±18% (pocket: 0.0318 → 0.0317). Not overfit to one
  tree mix.

**Value screen for `pocket`** (1M heldout, same statistics the richer
study's census used to admit bits; artifact
`updloc-artifacts/pocket-value-census.txt`):

| bit | P(b=1) | MI(t;b) bits | both cells ≥100 | top-8192 median min-cell |
|---|---:|---:|---:|---:|
| **pocket** | 0.304 | **0.0857** | 9,053 (90.8% mass) | **0.379** |
| rank1 | 0.613 | 0.1051 | 8,576 (90.2%) | 0.306 |
| null (hash) | 0.463 | 0.0022 | 10,944 | 0.487 |

MI 39× the content-free control, co-occupancy *better* than rank1: the
learned bit carries real, non-degenerate conditioning information. Cost ×
value together: **~82% of rank1's MI at ~60% of its flip price, in a
channel whose flip price is itself 2.5–7.7× cheaper than previously
believed.**

## 4. The budget line (prices both ways; artifact `census_wl50.txt`)

Per-unit prices, loaded C=4,514 (quiet C=4,100 in parens); "blended"
includes per-call accumulator-stream overhead, "marginal" is the
loads-model row share (i8 row = 40 L1d events, i16 acc read = 80):

| unit | blended | marginal | as addition |
|---|---:|---:|---:|
| +1 threat fold row/node | 142.5 (129.4) cyc | 99.3 | −5.4 Elo |
| +1 HalfKA fold row/node | 278.8 (253.2) cyc | 148.5 | −10.4 Elo |
| +1 Finny delta row/node | 419.1 (380.7) cyc | 310.6 | −15.4 Elo |
| +0.01 threat rebuilds/node | 16.4 (14.9) cyc | — | **−0.63 Elo** |
| one threat rebuild | 1,635 (1,485) cyc | — | — |

A candidate representation with e events/move, a active rows/position and
f context-flips/node prices, as an **addition**:
`Δc ≈ e × 2 × 0.61 × c_row + f × 0.578 × rebuild(a) + cache-term`, and as a
**replacement** it credits back what it displaces. The threat channel's
named symbols are **22.3% of cycles** (study-36 shares; 21.56% without the
rebuild row — the coordinator independently reproduced +42.0 Elo from that
figure; 19.4% on the b40 profile) ⇒ full displacement earns **+38…+44 Elo
of speed** before the recode POC's cache term (−1.5% quiet, more loaded).

**This reframes the thesis: a learned replacement for the threat channel
does not have to BEAT the handcrafted channel on value — it has to come
within ~40 Elo of it, because that is the speed it frees.** A learned
channel that matches FullThreats' value at half its update cost banks
~20 Elo; at zero cost, ~40.

Context-bit REPLACEMENT speed, corrected: rankhalf-for-mirror =
−0.00974 crossings/node × s × 1,635 cyc = −9.2 cyc/node ⇒ **+0.35 Elo**
(supersedes both the coordinator's hand +2.5-Elo estimate and any
at_full-frame figure ≈ +3…+5). The exact-crossing and attribution
corrections cut in the same direction: at true prices, context-bit
replacement gains are small ABSOLUTES (≤ ~0.4 Elo) because the thing being
replaced was never expensive — the real currency of a better partition is
conditioning capacity per flip (§3), not speed.

**The missing number, named:** FullThreats' VALUE in ARCHON has never been
measured (external anchors: SF's merge ~+30 fixed-node; PlentyChess +45.5
fixed-node). The engine already supports empty threat weights
(`apply_at`'s HalfKA-only branch), so the yardstick is a D48X-recipe
HalfKA-only twin — one full retrain (~14 h GPU), pre-registered here as
the arbiter any displacement decision needs. Until it runs, replacement
budgets are bounded by external engines' numbers, not ours.

## 5. Successor pairs are free — the locality objective's data path, validated

Every binpack-derived record already carries the played move
(`extra[0..2]`; 100% of the training mix's binpack corpora and both
heldouts), and the D46 quiet filter guarantees it is Normal, non-capture,
non-promotion — so the successor position is a pure relocation
synthesizable at the loader with **no new corpus, no new filter site, no
D46 exposure** (`tools/updloc/pair_seam.py`). Validated on 200k heldout
pairs: 100% valid, king-mover fraction 20.6%, and the per-perspective
index deltas through the trainer's own mappers:

| rows changed per played move (per perspective) | HalfKA | FullThreats |
|---|---|---|
| non-king moves | **exactly 2** (p99 = max = 2) | 3.92 (p90 8, p99 13) |
| king moves, mover's perspective | 21.4 (bucket rewrite) | 1.28 (p50 **0**) |
| king moves, opponent's perspective | 2.0 | 0.07 |
| all pairs | 4.00 | 3.25 |

The exactly-2 column proves the frame alignment (any orientation bug would
explode it); the p50=0 column is the threat channel's king-blindness made
visible — a quiet king move that stays on its mirror half changes zero
threat rows, because kings are neither attackers nor victims in the
SFNNv14-era space. Mean total update: **14.5 rows/move across both
channels and perspectives** — the number a learned code trained under a
locality penalty has to beat at equal value.

## 6. Tier-1 — the trained-properties panel (pre-registered, built, UNARMED)

Four arms on the proven 12-sb project-and-finetune recipe (D48X mix/frame,
same seed, identity baselines reused; specs in `tools/nnue/updloc_panel/`;
launcher written and **deliberately not armed**: the GPU is committed to
the D49 trainer for ~11 h at write-up time, long GPU runs start on the
owner's say-so, and the richer-representation panel has queue priority if
armed. The decision rules below were fixed BEFORE any arm has run and are
not modified after readout):

| arm | what it trains | what it measures |
|---|---|---|
| `refine_pocket` | the LEARNED context bit as a constant-row (tuple×bit) code2d refinement | does learned-locality conditioning carry heldout value at its (now ~free) engine price? |
| `refine_pocket_null` | same code, hash-bit region | content vs split-capacity |
| `ftsparse_a` (λ=3e-3, 2-sb ramp) | element-wise post-CReLU L1 — **never tried at a density-moving dose** (λ=1e-3 never moved density; default-0 is an admin ruling, not a verdict) | the heldout-per-density exchange curve |
| `ftsparse_b` (λ=1e-2, 2-sb ramp) | dose ladder point | — |

Validity gates: every arm's sb0 heldout must print 0.006350 (bit-parity;
the pocket seam was CPU-smoke-verified — identity code2d through the
pocket region reproduces the plain stream exactly; the ftsparse ramp
starts at 0 so sb0 parity holds trivially). Decision rules U1–U4 and the
slope guard are fixed in the launcher header (gate 25e-6; deployability
bar 32e-6 = the pocket bit's measured price at the *unfavorable* exchange
end — versus the 150e-6 bar rank1 faced under the refuted pricing: the
correction moves the bar ~5×). Cost: ~3.7 h GPU.

**The flagship follow-up, specified not built (L3):** learn the context
partition *jointly* with the net — soft assignment π: 64 king squares → C
cells; loss = task + λ_flip · Σ_{f,t} KTRANŜ[f,t]·(1 − ⟨π(f), π(t)⟩)
+ λ_act · E|A|, straight-through hardened, with λ set from this study's
measured prices (1,635 cyc/flip → Elo → heldout via the exchange rate) so
the optimizer trades value against exactly what the engine pays. The
KTRANS artifact is committed; the pair seam (§5) supplies successor pairs
if the penalty is taken per-move rather than per-matrix. Machinery
precedents: hash embeddings / DSelect-k (design study §4.2). Needs trainer
graph work + GPU — post-D49.

## 7. What landed (all committed, branch `learned-repr-study`)

1. `962fb798` — the four counter families (feature-gated; neutrality gate:
   eval hash + d13 pin exact).
2. `6759934e` — `tools/updloc/updloc_census.py` (+ partition annealer) and
   the artifact directory (dumps, battery, b40 symbol table).
3. `760b3702` — the pocket region bit (worktree trainer copy; live D49
   trainer untouched), `pair_seam.py`, `pocket_value_census.py`,
   `updloc_partition.py`, four arm specs + unarmed launcher, the
   `ft_sparsity_warmup` plumbing.
4. This document + regenerated corrected artifacts.

### 7.1 Gate log

`CHECK_NNUE_GATE=1 bash tools/check.sh` on the branch (both builds inside
the shared build flock + 6G systemd scope), actual output:

```
  eval hash  : 2e1ede9521624dd89eab4eb1bae0a1ea568cd6cf7e85b668179a7b5e93cf79c4
  d13 nodes  : 1739588
Bench: 587683
All checks passed.
```

All three pins bit-exact — the counter additions are behind `foldcount`,
off in default and release builds, as required. (Re-run after the
merge-from-main; result in the merge commit.)

### 7.2 Reproduction

```bash
# instrument build (any label; flock + cgroup discipline)
flock /home/nick/.cache/archon-build.lock \
  bash tools/perf34/build34.sh updloc "foldcount,sota,arch-d48x" native
# the two census frames (run from a checkout that has data/)
ARCHON_FOLDCOUNT=updloc_wl50.txt bin34/updloc \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
ARCHON_FOLDCOUNT=updloc_d13.txt bin34/updloc \
  bench 16 1 13 tools/nnue/calib/wl50.fens depth engines/canonical/net.cenn
# census + budget line, learned partitions, pair seam, value census
python3 tools/updloc/updloc_census.py updloc_wl50.txt
.venv/bin/python tools/updloc/updloc_partition.py updloc_wl50.txt \
  data/nnue/heldout/uho-band0-5.bin
.venv/bin/python tools/updloc/pair_seam.py
.venv/bin/python tools/updloc/pocket_value_census.py
# denominator battery (canonical PGO binary)
bash <scratch>/denom_battery.sh   # archived: updloc-artifacts/denom_battery.tsv
```

Neutrality gate for any counter change: the instrumented build must
reproduce the wl50 eval hash and the d13 node pin exactly, and
`make.calls` must equal the bench node total.

## 8. Assumptions, explicitly

1. **Study-36 symbol shares** transport to this frame (cross-checked
   against the b40 profile: big symbols agree to ±30% relative; the
   attribution correction is a 7–10× effect and robust to this).
2. **The survival factor s = 0.578** (wl50) / 0.566 (d13) is
   boundary-agnostic — the deferral machinery doesn't inspect the bit.
   Same assumption the richer study made; its §8.2 caveat stands.
3. **~120 Elo/speed-doubling** and **12–28e-6 heldout/Elo** are inherited
   project constants.
4. **Data-side occupancy from heldout band0-5** proxies the mix (the full
   mix census re-run is a launch-checklist item).
5. The annealer is heuristic; at 64 nodes with 3–4 seeds agreeing, treated
   as near-optimal.
6. wl50's tree mix is endgame-tilted (mean 12.9 pieces at eval; zero
   castling) — in-tree rates carry that tilt; d13 cross-checks all
   headline rates within ±12%.

## 9. What this study could NOT establish

1. **FullThreats' value in ARCHON** — the HalfKA-only yardstick twin is
   named and specified (§4) but needs a full retrain. Without it, every
   "replacement banks +40 Elo" line is a speed fact, not a net-Elo fact.
2. **The middlegame rebuild price** — bounded (≤~5.1k cycles via the
   eager-era receipt), not measured per-bit. A `pocket`-boundary-
   conditioned rebuild census would pin it (one more counter, same
   pattern).
3. **Whether pocket-conditioned threat rows carry heldout value** — that
   is exactly U1/U2, unarmed by design. Tier-0 establishes necessary
   conditions (MI, co-occupancy) and price only.
4. **Element-wise ft_sparsity's exchange rate** — U4's question; the prior
   λ=1e-3 arm is uninformative (density never moved), which is why the
   ladder doses 3× and 10× higher.
5. **The L3 joint objective's trainability** — specified with its data
   assets in place, not smoke-run (trainer graph work gated on GPU).
6. **Study 36's per-node totals** — refuted (again), but *why* its
   denominator was ~1.84× high was not diagnosed here.

## 10. Retractions and corrections in this session

1. **Retracted (own, in-session):** the first learned-partition readout
   (a "14.8× cheaper castled pocket at 20% mass") — built on a
   double-flipped `opp_ksq`; the corrected numbers (§3) are 1.4–3.2×, not
   14.8×, and the corrected pocket is {g1,h1,h2} at ≥30% mass. Caught by
   re-deriving the occupancy against decoded boards before writing this
   doc.
2. **Corrected (predecessor's):** the richer-representation study's
   context-bit price table (§2 above) — `at_full` misattribution, 2.5–7.7×
   overpriced. Its qualitative ordering (head door free < HalfKA < threats)
   survives; its quantitative verdict ("−6…−14 Elo per bit") does not.
3. **Confirmed (coordinator's):** study 40's cycles/node over study 36's,
   by an independent 7-round startup-subtracted battery.
