# A richer representation at constant cost — the refinement study (2026-07-31)

**Status:** design + Tier-0 pre-screen, all measured; Tier-1 panel built,
smoke-verified end-to-end on CPU, specs committed, **not launched** (the GPU
queue at study time was identity2 → d49x ~13 h; launching is the owner's
call — `tools/nnue/refine_panel/recode_t1_refine_launch.sh` is written but
deliberately not armed). Predecessors:
`docs/research/learned-feature-code-study-2026-07-30.md` (the design study)
and `docs/research/recode-poc-2026-07-31.md` + the Tier-1 compression panel
(b373c98), whose verdict this study builds on: **the learned code beat the
content-free control 2.3× (kmeans8k +165e-6 vs random8k +383e-6 at sb12,
identity 0.006410) but compression cannot buy Elo, because the fold work is
`active_features × ACC` and neither term moves.**

**Verdict in one paragraph.** The commissioning sketch — "expand the base
vocabulary with finer geometric distinctions, then learn a surjection back
down" — is, taken literally, **vacuous**: both index functions are already
*injective* on their atom spaces (§1), so a finer function of the same atoms
adds zero information and a surjection over a redundant expansion is
relabeling. New information can only enter through three doors, and this
study measured the price of each (§2–§3): **(a) input-side context
refinement** (slow king-state bits conditioning threat rows at constant row
count, funded by the 11,170-row dead budget) is possible and information-rich
(census §4) but **not free — every candidate bit costs a measured 3.4–8.6%
of cycles (−6…−14 Elo) in threat-rebuild rate**, because threats have no
Finny cache and a context flip is a monolithic rebuild; **(b) HalfKA-side
enemy-king context** is 4.7× cheaper per flip (Finny-cached) but has a thin
dead budget; **(c) head-side region-conditioned output buckets are the one
genuinely ~zero-cost door.** The pre-screen therefore prices the *value*
side with a four-arm Tier-1 panel (oracle upper bound, head-side arm,
constant-row candidate, content-free control), pre-registered in §6, with
every arm's sb0 verified bit-identical to the incumbent — the panel decides
which door, if any, clears its own measured bar.

---

## 1. The cost accounting — what is actually free, with receipts

Per position the engine pays `active_features × ACC` in the fold, plus
`refresh_rate × rebuild_cost` per channel, plus table-size cache effects.
The POC measured the cache term (9.5× shrink → 1.5% cycles quiet): small.
The three ways to make the representation *richer*, priced:

**1.1 Finer functions of the existing atoms: zero information.** Both codes
are injective on their (atom, context) domains — pinned facts, not
assumptions: `threats_active_in_range_and_distinct`
(`src/eval/nnue/sota_features.rs:1108`) proves the threat index is
collision-free over its 60,720 tuples (attacker piece incl. color, from, to,
victim piece incl. color, orientation applied, victim color in the index via
`atd_color * num_types + map`); HalfKA at B=32-with-mirror is a bijection on
mirrored king squares (design study §3.1). An injective code already
distinguishes everything its inputs distinguish; "finer geometric
distinctions" of the same tuple are functions of the tuple — coarsenings,
never refinements. **The brief's sketch only becomes real when the base
space gains a NEW input.**

**1.2 New atom families (more relations firing): linear, priced, not this
study.** Each new firing atom raises `active_features` (SFNNv16's PP_3Wide
et al. — the design study's §6.3 arms). Real, but it is exactly the
"must be earned" row of the commissioning cost model. Nothing here is free.

**1.3 New *context* (slow king/board state conditioning existing rows):
the actual opening — with a measured, non-zero price.** A context bit does
not change how many features fire or how many rows exist (splits funded by
dead rows, §4). Its entire price is the rebuild triggered when the bit
flips. Measured this session (foldcount `kmove` counters, d13 bench, wl50,
1,739,588 nodes, canonical D48X net):

| trigger | flips/node | est. rebuilds/node¹ | Δcycles² | ≈ Elo³ |
|---|---:|---:|---:|---:|
| d/e mirror (the existing context; calibration) | 0.03345 | 0.01892 (measured: `thr_refresh.calls`) | (in baseline) | — |
| own back-rank vs off it (`rank1`) | 0.05094 | 0.0288 | +7.4% | −12.4 |
| own back-two vs advanced (`adv`, = `rank12` boundary) | 0.05278 | 0.0299 | +7.7% | −12.8 |
| file-pair beyond mirror (`fpair`, added) | 0.05939 | 0.0336 | +8.6% | −14.3 |
| board half (`rankhalf`) | 0.02329 | 0.0132 | +3.4% | −5.8 |
| enemy-side bits (`esame`; enemy king's d/e) | 0.03345 | 0.0189 | +4.9% | −8.3 |

¹ flips × the measured deferral-survival factor s = `thr_refresh.calls` /
mirror-flips = 32,914 / 58,185 = **0.566** on the same run (the deferral
machinery is boundary-agnostic; §8 lists this as an assumption).
² at the perf-study-36 price `at_full` = 5.50% cycles at 0.0214/node ⇒
2.57% per 0.01/node. ³ at ~120 Elo/speed-doubling.

The counters exist because in-tree king behavior is not game king behavior,
and indeed **search crosses rank boundaries more often than d/e** (back-rank
shuffles, Kh1 tucks, castling lines) — a fact that kills the naive "kings
rarely advance, so `adv` is nearly free" argument I would otherwise have
written down. There is no threat-side Finny escape: the active *relation*
set is unchanged by a bit flip, but the engine keeps no per-ply active-row
list, so re-enumeration ≈ `at_full` regardless; and splitting only rare
tuples to cheapen the flip caps the value by the same construction.

**1.4 The same context on the HalfKA side: 4.7× cheaper per flip.** HalfKA
refreshes are Finny-cached: `finny_refresh` = 8.75% cycles at 0.1610/node ⇒
**0.54% per 0.01/node** vs threats' 2.57%. An enemy-king bit (HalfKA is
enemy-king-blind) at `esame` rates ⇒ ~+1.8% cycles ≈ −3.1 Elo, plus Finny
memory ×2 and worse slot staleness. Cheaper, not free; thin dead budget
(~2.2k of 22,528 rows) caps constant-row splits at ~10% of the table.

**1.5 Head-side conditioning: the free door.** The output-bucket index is
already a learned-partition slot evaluated O(1)/node. Extending it from
`layout[piece_count]` (8 buckets, a handcrafted scheme) to
`layout[piece_count] + 8·b(king-region)` doubles the head tables (a few
hundred KB against a 68 MB net) and adds **zero** refresh, zero fold work,
zero per-event cost. It conditions the *readout* of the whole accumulator
rather than individual rows — a different (coarser) function class than
input-side splits, which is why the panel carries both.

**Summary of the accounting:** compression (the POC) and refinement (this
study) are duals that fail free-lunch for dual reasons — compression cannot
touch either fold term; input-side refinement pays at the *refresh* term,
which the design study's F2 had priced only for the full-bucket case. What
is genuinely ~free: the code *content* (POC: real, 2.3× vs null) and the
*head-side* context. Everything else has a measured meter running.

## 2. Where the freedom actually is

The one genuinely under-conditioned channel is threats: **king-blind except
the 1-bit mirror** (verified: `orient_threats` is the only king dependence),
while HalfKA is fully own-king-conditioned (the probe confirmed this
structurally: own-king bits are *degenerate* within HalfKA rows —
`tuples_ok=0` — exactly as a full-res own-king context predicts) but
enemy-king-blind. A linear FT cannot represent king × threat-relation
interactions unless the code carries them. So the refinement question,
concretely: **do threat-row values differ by king region enough to pay the
measured refresh price — or, failing that, enough to be worth capturing
through the free head-side door?**

The capacity to condition at constant cost exists and is large:

- **Dead budget** (census, §4): 11,170 of 60,720 threat rows (18.4%) never
  fire across the 8M-position training-mix sample ∪ the 1M heldout.
- **The built constant-row code** (`tools/nnue/refine_code.py`):
  11,160 king-split tuples carrying **93.6% of all threat fire mass**, at
  exactly 60,720 rows — dead tuples merged into 10 per-attacker-family
  scrap rows (blunder-tail guard: a rare tuple falls back to its own
  family's value, never a global average).

## 3. What was built and measured this session (all committed)

1. **`tools/nnue/region_census.py`** — the joint (threat-tuple ×
   king-region-bit) census, mix-weighted over the 12-pack D48X training mix
   (8M) + full heldout (1M). Region bits (canonical definitions now in
   `archon_jax/features.py::region_bit`, one source for census, probe, and
   loader): `adv`, `rank1`, `fpair`, `esame`, `eadv`, `null` (position-hash
   control). Results in §4.
2. **`foldcount` king-boundary counters** (`kmove.*`) + a hook in
   `king_cross` — the in-tree crossing-rate instrument behind the §1.3
   table. Feature-gated; default and release builds untouched; fmt/clippy
   clean.
3. **`tools/nnue/region_probe.py`** — the frozen-incumbent first-order
   split-gain probe: one vjp through `head_forward` yields the per-position
   accumulator gradient, which IS the shared gradient of every active row;
   scattered by (tuple, bit) it prices a split before any training, with an
   analytic debias (per-cell second moments) AND the empirical hash-bit
   null. Forward parity vs `model.forward` asserted exact. Results in §5.
4. **The trainer seam** (`features.py` `threat_code2d`/`bucket_regions`,
   `spec.py`, `data.py`, `recode_tier1.py` `project_threat_code2d` +
   `widen_head_buckets`) — zero graph changes for the input-side arms; the
   head arm reuses `num_buckets`. Verified end-to-end on CPU
   (`validate_refine_seam.py` results, §6.1).
5. **`tools/nnue/refine_code.py`** — deterministic code builders (refine /
   oracle) consuming census + probe artifacts, emitting manifest + sha256.
6. **Four arm specs** (`tools/nnue/refine_panel/recode_t1_{oracle_rank1,
   headbucket_rank1,refine_rank1,refine_null}.json`) + the unarmed launcher.

## 4. Tier-0a — the joint census (necessary conditions: PASS)

9M positions (8M mix + 1M heldout), both perspectives:

| bit | P(b=1) | MI(t;b) bits | tuples with both cells ≥100 | mass | top-8192 median min-cell share |
|---|---:|---:|---:|---:|---:|
| `adv` | .104 | .0856 | 16,494 | 94.3% | .080 |
| `rank1` | .319 | **.1213** | 24,322 | 98.6% | **.297** |
| `fpair` | .305 | .0536 | 23,031 | 98.5% | .283 |
| `esame` | .790 | .0529 | 22,256 | 98.3% | .222 |
| `eadv` | .104 | .0827 | 16,710 | 94.4% | .080 |
| `null` | .500 | .0002 | 28,038 | 99.3% | .497 |

Every candidate carries 250–600× the null's MI; hot-tuple co-occupancy is
real (a `rank1` split of a median top-8192 tuple has 30% of its firings in
the smaller cell — these are not degenerate conditionals). Band caveat:
2,570 rows fired only in the mix sample and 68 only in heldout — the dead
set is defined on the union, and games remain the final guard.

## 5. Tier-0b — the gradient probe (the value screen)

READOUT PENDING AT COMMIT TIME — the 393k-position run was mid-flight; this
section is filled by the follow-up commit with the per-bit debiased gains
vs the null bit, and the chosen bit for the panel. Pre-registered use: the
probe RANKS the candidate bits and selects split tuples; it does not gate
the oracle arm (one inference deep — census co-occupancy alone justifies
measuring the oracle), and a flat probe with a passing oracle is reported
as a probe failure, not a value failure.

(Method validation already in hand from the smoke run: forward parity
exact; own-king bits degenerate on HalfKA rows as predicted; the debias
correctly drives small-n gains to ≈0 at n=2k.)

## 6. The Tier-1 refinement panel — pre-registered

Four arms on the existing panel recipe (12 sb, LR 1.1e-4, D48X mix/frame,
same seed), reusing the ALREADY-RUN identity baselines (identity 0.006410,
identity2 0.006427 at sb12 — the seed-repeat pins per-readout noise):

| arm | rows | what it measures |
|---|---|---|
| `oracle_rank1` | 77,104 | value upper bound: top-16,384 tuples (96.9% mass) get a dedicated per-bit row; trainer-only (u16 TRM cannot hold >65,535 rows) |
| `headbucket_rank1` | 60,720 (16 head buckets) | the free door: king-region output buckets |
| `refine_rank1` | 60,720 | the deployable-shaped candidate: 11,160 splits at constant rows |
| `refine_null` | 60,720 | the null: SAME code, SAME init, hash-bit conditioning — isolates content from split-capacity/optimizer effects |

### 6.1 Validity gates (already verified where possible)

Every arm must print **sb0 heldout = 0.006350** — the refined init is
conservative (both cells start at the incumbent row; merged rows never fire
on mix∪heldout), so sb0 is bit-parity, and this was **verified on CPU:
max|Δ| = 0.0 for refine_rank1, refine_null, and headbucket vs the incumbent
on heldout records, while the refined stream differs on 47% of active slots
(= split share 92.8% × bit-difference 50.6%, decomposition checked), and
one training step runs for both arch shapes.** A different sb0 voids that
arm. (The pallas GPU path is exercised only at launch; the reference path
is what was smoke-verified.)

### 6.2 Decision rule (pre-registered 2026-07-31, before any arm has run —
do not modify after readout)

Statistic: **m(arm) = mean heldout over sb8–12** (per-readout paired noise
between the two identity runs: sd 16.3e-6, sb12 |Δ| 17e-6; the sb8–12
means differ by 2.4e-6; the gate below uses 25e-6 ≈ 3× the estimated
tail-mean noise, conservative against the lucky-tail-mean read).

- **R1 (does king-conditioning carry value at all):**
  m(oracle) ≤ m(identity) − 25e-6. Fail ⇒ input-side refinement is a
  measured null at this budget; report it; R4 still read.
- **R2 (content, not structure):** m(refine_rank1) ≤ m(refine_null) − 25e-6.
- **R3 (input-side deployability):** m(identity) − m(refine_rank1) ≥
  **150e-6** — the refresh price of `rank1` (−12.4 Elo, §1.3) at the
  favorable end of the project's heldout↔Elo exchange (~12–28e-6/Elo, the
  same pricing the compression verdict used). **Expected NOT to clear**;
  the arm is science (the retention fraction oracle→constant-rows) unless
  the gain is startling.
- **R4 (head-side deployability):** m(headbucket) ≤ m(identity) − 25e-6 ⇒
  the head arm is the deployment candidate (its runtime cost is ~0);
  proceed to Tier-2: engine bucket-index change + full retrain + **games
  arbitrate** (the standing rule; the D48X-era eval/search confound applies
  to any strength read).
- **Slope guard (pre-registered, the only allowed extension):** if an arm
  is within 2× its gate of passing AND its sb8–12 slope is steeper than
  −5e-6/sb, ONE 24-sb rerun of that arm + identity is allowed. Nothing
  else changes after readout.
- **Bit choice:** `rank1` is the placeholder (highest MI, best co-occupancy);
  if the probe's debiased ranking overturns it decisively (>2× gain on
  another census-viable bit), the specs are regenerated for that bit BEFORE
  launch — a Tier-0 decision, never a post-readout one.

Cost: 4 × ~55 min ≈ 3.7 h GPU, queued behind d49x on the owner's go.

### 6.3 What Tier-2 needs if a gate passes (deliberately unbuilt)

Head door: engine bucket index = f(piece_count, region bit) + 16-bucket CENN
head sections (~small, O(1)/node); code2d door: a TRM2 (tuple,bit)→u16
section (243 KB), the bit in the resolver key, and a rebuild trigger on the
bit's boundary (~50 lines at the same chokepoint TRM1 used) + gate re-pins.
Either way: full retrain, ledger rows for (net, code, region) — the code and
its region bit are now part of a net's identity — then games.

## 7. What would make this fail (named in advance)

1. **The 12-sb finetune may be too short for refinement specifically.**
   Compression recovery *descends from a dent*; refinement must *grow a
   distinction from an exactly-flat start* at LR 1.1e-4. A null panel can
   mean "no value" or "no budget" — the slope guard is the only
   pre-registered probe of the difference, and an all-flat oracle WITH a
   flat slope is reported as a genuine null at this budget, not stretched.
2. **Heldout ≠ Elo, and the era is confounded** (canonical net promoted
   while losing). Games arbitrate any deployment; expect wide bars.
3. **The refresh pricing carries two transported constants** — the 5.50%
   `at_full` share (perf-study-36 config) and the 0.566 survival factor
   (this bench's deferral realization). Both are same-engine, same-net
   measurements, but a Tier-2 candidate must re-price on its own binary
   (the POC's speed-AB frame does exactly this).
4. **Probe myopia:** first-order, frozen trunk, one corpus band
   (uho-band75-100 positions 4M+), top-4096 tuples. It ranks; it cannot
   veto the oracle.
5. **Head-bucket data dilution:** 16 buckets halve per-bucket head data —
   the folklore failure mode of bucket proliferation. The arm's warm start
   (duplicated buckets) means sb0 is safe; dilution would show as a
   *worsening* trajectory, which the rule reads as a fail, correctly.
6. **OOM discipline:** the first probe run and the seam validation were
   both OOM-killed when run concurrently (exit 137, ~250k positions lost —
   the probe now checkpoints every 128k). One heavy CPU job at a time
   beside the trainer; `free -m` before builds; the oracle arm's +250MB
   GPU footprint fits (identity2 held 6,004 of 8,188 MiB) but was not
   measured under pallas at 77k rows.

## 8. What I am least sure about, in order

1. **Whether the finetune budget can differentiate split rows at all** —
   the flat-start asymmetry in §7.1 is the biggest threat to reading a
   negative panel correctly. If the oracle fails R1 with a steep slope, the
   honest answer is one pre-registered extension, then acceptance.
2. **The transported refresh constants** (survival factor per boundary,
   `at_full` share per config). A factor-2 error moves every §1.3 Elo
   price by a factor 2 — it would not change the *ordering* (head door
   free, halfka cheaper, threats dear) but would move R3's 150e-6 bar.
3. **The heldout↔Elo exchange rate** (12–28e-6/Elo) — inherited from the
   compression verdict's pricing; D49 showed the heldout map can
   over-predict match Elo by ~1.3–2.5×. R3 uses the favorable end
   deliberately (an arm that cannot clear even that is dead); a passing R3
   would still need games.
4. **Whether `rank1` is the right bit** — highest MI and co-occupancy, but
   the probe (pending at commit) may rank `esame` or `adv` higher on
   gradient divergence; the pre-launch regeneration rule covers this, but
   a wrong placeholder that survives a lazy launch would waste the panel.
5. **The scrap-row tail risk** — merged never-fired rows CAN fire in games
   (they never fire in 9M training/heldout positions, but band coverage is
   not game coverage). Per-family scrap + the eventual game gate are the
   mitigations; a Tier-2 candidate should also run the tactical-FEN Δeval
   screen the design study specified (T0b) before any match.
