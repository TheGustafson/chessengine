# Learning the feature code — can the sparse binary index function be learned? (2026-07-30)

**Status:** research & design study (no training runs launched; D49 was live on the
box). **Deliverable:** the freedom analysis, prior art (engine + ML), a staged
buildable proposal ("Recode"), and a pre-screen that answers in hours. One
Tier-0 instrument was built and run as part of this study
(`tools/nnue/feature_census.py`, results in §5.1).

**The question, precisely.** ARCHON already *learns the embedding*: the
feature-transformer table `W[feature, lane]` is trained end to end. What is
handcrafted is the **sparse binary index function** — `halfka_index()`
(22,528 dims) and `threat_index()` (60,720 dims) in
`src/eval/nnue/sota_features.rs` — the code that decides *which rows fire*.
This study asks whether that code can be learned, subject to the constraint
that makes an NNUE an NNUE: a move must change O(1) features so the
accumulator updates incrementally.

**Verdict in one paragraph.** The constraint binds *softer* than the naive
reading in one decisive way and *harder* in another. Softer: because the
(atom, context) input space of each channel is **finite and already routed
through lookup tables**, the content of the code is completely unconstrained —
any learned mapping compiles to table entries, and the engine's incremental
machinery is untouched. The largest freedom is therefore not the vocabulary or
the conditioning but **the mapping itself**, which can be learned trainer-side
and deployed as data. Harder: firing *more* rows per atom (k-hot codes) is
priced by the fold kernels, which are already the two largest cycle consumers
in the engine, so k>1 proposals need ~20+ Elo of eval gain per channel to
break even. The ML literature (§4) confirms the constraint is a
**turnstile-sketch/homomorphism requirement, not an LSH-locality one**, and
that learning a code under an explicit update-cost constraint is an unstudied
problem — the publishable angle. The proposal (§6) is a staged program whose
first arm changes zero engine lines, and the pre-screen (§5) gives a
learnability verdict in ~4-5 GPU-hours, with Tier-0 numbers already measured.

---

## 1. What the engine actually computes (facts, with sources)

Per perspective `p`, the accumulator is

```
acc_p = ft_bias + Σ_{t ∈ A_p(pos)} W[t]          (i16 lanes, ACC wide)
```

where `A_p(pos)` is the active-feature set, produced by two channels:

| | HalfKAv2_hm | FullThreats |
|---|---|---|
| dims | 22,528 | 60,720 |
| atom | (piece, square) | directed attack relation (attacker, from, to, victim) |
| context | king bucket (32) × mirror (2) per perspective | mirror half only (1 bit) |
| active/pos (measured, band75-100) | 14.4 mean, ≤32 | 12.4 mean, ≤128 |
| Δ per move | ≤4 (2 sub / 2 add) | ≤80 events (x-ray ripple) |
| full-refresh rate | 0.1610/node (any own-king move; Finny-cached) | 0.0214/node (mirror flip only; plain rebuild) |
| weights | i8 (perf study 32) | i8 |
| index machinery | closed-form + `KING_BUCKETS` LUT | three LUTs (`lut1`, `offsets`, `lut2`) |

(Version note: our 60,720-dim threat layout is the SFNNv14-era space the D40
transcription pinned; SF has since moved to 59,808 (SFNNv16, §3.1) — the
reference is in motion on exactly the axis this study is about.)

Sources: `src/eval/nnue/sota_features.rs` (index math, LUT construction),
`src/eval/nnue/sota_threats.rs` (the ≤80-event dirty-diff observer; the
"mirror refresh" note documents that a threat index depends on the king ONLY
through its mirror half — the 1-bit context is *why* the threat refresh rate
is 7.5× lower than HalfKA's), `design/11-evaluation.md` §12.1-12.2, and the
measured rates from the perf-study series. The fold path narrows to u8 at
activation (`ft_shift = 9`) and exploits `find_nnz` sparsity at the L1
boundary; an i8 *accumulator* is arithmetically impossible (measured
2026-07-30: the bias alone is 11.7× the i8 range).

Cost anchors (perf study 36, within-run attribution shares, D48X @ native):

| symbol | %cycles |
|---|---:|
| `avx512::i8_deltas_into` (HalfKA fold) | 13.77 |
| `avx512::i8_deltas_into_generic` (threat fold) | 12.38 |
| `HalfkaContext::finny_refresh_into` | 8.75 |
| `SotaHead::forward_into` | 6.15 |
| `ThreatContext::at_full` (threat from-scratch) | 5.50 |
| observer + apply + ensure_threats | ~4 |

The input-representation machinery is **~45% of all engine cycles**, and the
weight tables are far out of cache: at ACC = 1280, HalfKA is 22,528 × 1280 i8
≈ 28.8 MB and threats 60,720 × 1280 i8 ≈ **77.7 MB** against a 32 MB L3
(7700X). The fold kernels' cache misses are the engine's largest miss
population (study 29 census, prior tree). This matters below: **shrinking the
threat row space is a speed lever, not just a data-efficiency lever.**

---

## 2. The freedom analysis — what the constraint actually pins

### 2.1 The constraint, stated exactly

Incremental maintenance requires that per-atom contributions combine under an
**abelian group operation with a cheap inverse** — you must be able to
*subtract* a departing atom. Sums over Z^ACC are the only NN-useful instance
(XOR gives Zobrist hashing; max is not invertible, which is why no
max-pooled NNUE exists). In streaming terms the accumulator is a **linear
sketch in the turnstile model**: inserts and deletes of atoms, exact O(|Δ|)
maintenance. This is a *stronger* property than LSH locality ("similar inputs
→ similar codes") and a different one — a code can be perfectly
locality-sensitive and still be unmaintainable (MinHash cannot process
deletions). The ML survey (§4) confirms these two properties are routinely
conflated and must not be.

The full requirement on the active-set function decomposes as:

```
A_p(pos) = ⋃_{a ∈ S(pos)} C_p(a, ctx_p(pos))
```

- `S(pos)` — the **atom vocabulary**: any set-valued function of the position
  with a bounded, cheaply enumerable per-move symmetric difference.
  Piece-squares (|Δ| ≤ 4) and attack relations (|Δ| ≤ 80, via the x-ray
  observer) are the two instances built; the class is wider (any local
  predicate with bounded ripple), but each new family costs an observer.
- `ctx_p(pos)` — the **context**: any function of the position, provided its
  transition rate is low enough to pay a full rebuild when it changes. This is
  a *budgeted violation* of decomposability, not a prohibition: cost =
  rate × rebuild. Measured instances: HalfKA's (bucket, mirror) at 0.1610
  rebuilds/node (amortized by the Finny cache), threats' 1-bit mirror at
  0.0214/node.
- `C_p` — the **code**: any map from (atom, context) to a small set of rows.

Two exemptions are worth naming because they move work *outside* the
constraint entirely: (i) per-node O(1) global features (material, popcount
buckets, rule50) enter on the **head side** free of incrementality — the 8
output buckets are already a learned-partition slot; (ii) anything computable
at export time is free — the code's *training-time* form can be arbitrarily
expensive (a neural encoder, an optimization), because the finite domain
compiles to a LUT.

### 2.2 The five freedoms

**F1 — the code table itself (the finding the naive framing misses).** At
fixed vocabulary, context, and k, the map `C` is **completely unconstrained**,
because both channels' index functions are already table-driven
(`KING_BUCKETS`; `lut1`/`offsets`/`lut2`) and every (atom, context) tuple is
enumerable (HalfKA: 64 king-states × 11 piece-kinds × 64 squares before
mirror-dedup; threats: the 60,720 relation tuples). Merging rows, splitting
rows (up to context granularity), aliasing, re-allocating — all of it is
LUT content. **Runtime cost at fixed k and row count: zero. With fewer rows:
negative** (a 60,720 → 8,192 threat re-code shrinks the table 77.7 MB →
10.5 MB, into L3 territory). This is where "learn the code" lives, and the
incremental-update constraint does not touch it.

**F2 — the context partition.** The granularity of `ctx` is a measurable-cost
dial with two existing data points (0.1610 vs 0.0214/node). Key asymmetries:
- HalfKA refreshes on **every own-king move** regardless of bucket table
  content (design/11 §12.2), so *any* 64→B table is refresh-rate-identical —
  the layout is free at fixed B. Note (per §3.1) that at B = 32 with the
  mirror our table, like SF's, is **full resolution** (a bijection on the 32
  mirrored king squares) — so the learnable object is the *coarsening*
  (B < 32) and its layout, which trades data density and table size against
  specificity. Bucket count B additionally changes Finny staleness (fewer
  visits per slot → bigger set-diffs) and Finny memory, both measurable
  offline.
- Threats' 1-bit context is load-bearing: upgrading threats to bucketed
  conditioning would raise their refresh rate ~7.5× with no Finny cache;
  scaling `at_full`'s 5.50% cycle share by 7.5 prices it at ~40% of engine
  cycles — ruled out by arithmetic, no experiment needed.

**F3 — the atom vocabulary.** Learnable at two costs. Zero-engineering-cost:
the threat relation filter (`MAP`/`NUM_VALID_TARGETS`/`KEPT_VICTIMS` in
`sota_features.rs`) is a handcrafted feature-selection mask over attacker ×
victim type pairs — SF's choice, transcribed. Dropping low-value families
reduces events/move (speed); adding families the mask excludes is a mask flip
plus table rebuild on the *existing* observer (the observer over-generates
and filters). The most interesting excluded family: **king-as-victim
relations** (threats-on-king ≈ check/king-pressure structure; the current MAP
excludes kings as victims entirely). Non-zero-engineering-cost: genuinely new
atom families (pawn-structure predicates, defended-status) each need a
bounded dirty-diff observer — feasible (threats prove the class) but each is
an engine project.

**F4 — multiplicity k (rows per atom).** Nominally free, practically
expensive: fold cost is linear in events × k, and the two fold kernels are
already #1 and #2 in the cycle profile. Doubling one channel's fold work ≈
+13% cycles ≈ −11% NPS ≈ **−20 Elo** at the measured ~120 Elo/speed-doubling
constant. So k-hot codes (e.g. runtime factor rows, "HalfKA + A" kept live)
must buy 20+ Elo of eval quality per channel. Not impossible — but the burden
is quantified, and F1-style re-coding at k=1 is free by comparison.

**F5 — what is genuinely forbidden.** Row selection that depends on
fast-changing non-atom state (a per-position global encoder at the input — the
standard VQ-VAE shape); non-invertible aggregation (max/top-k over the whole
position); anything whose per-move delta is unbounded. Global top-k is the
crisp example: adding one atom can evict arbitrarily many winners, so a single
move rewrites an unbounded portion of the code. Top-k *per atom* is fine.

### 2.3 Corrections to the brief's reading

The brief's reading — learnable parts are (a) vocabulary, (b) context
conditioning, (c) atoms fired per position — is right but incomplete, and its
emphasis is inverted:

1. **Softer than stated:** the dominant freedom is F1, the code *mapping*
   itself, absent from the brief's list. It is unconstrained, deployable as
   data (LUT/table contents), and in the threat channel it carries a speed
   upside. A study that only explored vocabulary and conditioning would have
   left the cheapest lever untouched.
2. **Harder than stated:** "how many atoms fire" hides two dials with opposite
   prices. The *filter* (which atoms exist, F3) is cheap to change and can
   only reduce runtime cost; the *multiplicity* (k, F4) multiplies the
   engine's hottest kernels and is close to priced out at current NPS.

---

## 3. Prior art — chess engines

All Stockfish claims below were read from local commits in
`stockfish_reference/Stockfish` (which carries master refs through SFNNv16),
not from memory; engine-table claims from READMEs/source/talkchess as cited
in the survey. (Note: `integral_reference/` is referenced by CLAUDE.md but
the clone is absent on disk; Integral facts here are web-sourced.)

### 3.1 The Stockfish lineage — and what its history actually says

| step | commit | change | evidence |
|---|---|---|---|
| HalfKP 41,024 | `84f3e867` (2020-08) | (own-king-sq × piece × sq), from shogi | +92.8 ± 2.1 |
| HalfKAv2 45,056 | `e8d64af1` (2021-05) | kings become piece planes; flip not rotate; PSQT skip; 8 material subnets | +21.7 STC / +5.9 LTC |
| HalfKAv2_hm 22,528 | `d61d3858` (2021-07) | horizontal mirror halves the table — **and the freed budget bought L1 512→1024 in the same commit** | passed STC+LTC |
| FullThreats +79,856 | `8e5392d7` (2025-11) | threat relations added; L1 3072→1024 to pay | ~+30 fixed-node, 5-15% slower, +15 ARM, scales with TC |
| SFNNv11 66,864 | `fac506bd` (2026-01) | **piece→king threat rows REMOVED** — "13 MB and ~0.8 updates/move" saved | **non-regression** |
| SFNNv12 60,144 | `83e42045` (2026-02) | king-attacker rows removed | non-regression |
| SFNNv14 60,720 | `5eeca739` (2026-04) | pawn-push-vs-pawn rows added (credits Reckless/Stormphrax) | Elo-gainer STC+LTC |
| SFNNv16 59,808 | `f4bcd404` (2026-07) | PP_3Wide pawn-pair family added, pawn-push rows superseded | failed STC, passed LTC+ |

Three structural lessons, each load-bearing for this study:

1. **Feature-set moves are capacity trades.** Twice the freed budget from a
   smaller index space was spent on width *in the same commit* (hm → L1×2;
   threats → L1÷3). A learned compression (§6.1) should expect its win in
   the same currency: rows → width/speed, not rows → nothing.
2. **The post-merge threat history is a hand-executed walk-down**: whole
   semantic families (piece→king, king-attacker) shipped for months, were
   then deleted as *non-regressions*. Hand allocation demonstrably ships
   dead capacity; SF finds it by proposing one family at a time. A learned
   allocator searches that space wholesale — this is the strongest
   engine-side evidence that §6's program has something to find. It also
   sets a **negative prior on re-adding king-victim relations** (they were
   tried at scale and removing them cost nothing — §6.3 amended
   accordingly).
3. **The implementation decides the sign of a feature idea.** SF's first
   threat port (no incremental tracking) was **−90 Elo at +50% time odds**
   on an identical feature set; PlentyChess's bitboard incremental tracking
   turned the same idea into +45.5 fixed-node (−3.1 STC — quality paying
   only at depth). Update-cost engineering is not a tax on the
   representation question, it *is* the representation question — which is
   the engine-world's phrasing of our F4.

**The king-bucket table is not a clustering in SF.** Verified against
`half_ka_v2_hm.h` and `git log -G KingBuckets`: the 32 buckets are **full
resolution on the mirrored half-board** (one bucket per reachable mirrored
king square), never re-laid-out since introduction, and no fishtest attempt
to tune a layout was found. The *coarsening* question exists only below 32.

### 3.2 The rest of the field — hand-designed partitions everywhere

Input king-bucket counts in production: SF/PlentyChess-psq 32 (full-res),
Berserk 16, Viridithas 16, Obsidian 13, Integral 12 (factorized),
Reckless 10 (non-uniform: **8 of 10 buckets on ranks 1-2**), Motor 8,
Leorik 5, many at 1. Design regularities are folklore, not measurement:
resolution concentrates where kings live; bucket count is priced in net
size and data density ("king-bucket indexing wants ~1B positions" —
talkchess), with factorization as the standard mitigation. **No published
attempt to learn or systematically optimize a bucket layout exists** (named
searches in the survey came back empty). The bucket table is bullet's
`ChessBucketsMirrored` argument — a hand-written 32-entry array in every
engine that uses it.

Threat/relational inputs: originated in **Monty** (viren; +48/+36 there),
first AB port in Yukari (+30), SF as above, Reckless's independent 66,864
design (numerically identical to SFNNv11's space; same research community —
SF's SFNNv14 credits Reckless), Velvet. Reckless's asymmetries prefigure our
own: threat weights i8 vs psq i16, threat table = **81% of its 63 MB net**,
threats condition on the mirror bit only. Rejected variants on record:
x-ray threats, threats-*replacing*-psq ("greater than 50 Elo fixed nodes
loss in Monty").

### 3.3 Factorization — the training side already softens the code

nnue-pytorch's factorizer trains virtual features (HalfKP^ = HalfKP + P +
HalfRelativeKP) whose weights are summed into the real rows at export —
"even the rarest of feature weights being populated quickly with reasonable
values." The HalfKAv2_hm merge itself trained factorized. bullet makes it a
first-class type (`Factorised<A, B>`), and Integral's "768×12 factorized" is
exactly that. **Factorization is learned generalization over a
hand-specified equivalence relation** — the training pipeline already treats
the index function as something to be softened; nobody has yet let the data
choose the equivalence classes. That is precisely the F1 freedom, and (per
§4.2) it is structurally identical to the recsys world's
complementary-partition embeddings.

### 3.4 Learned sparse input codes: the null, with one open lead

No engine or paper that learns the sparse index function itself was found
(named searches recorded in the survey). Nearest misses: the factorizers
(hand-specified sharing), Monty's 768×4 threat-status conditioning
(handcrafted). One located-but-unread lead: the talkchess thread
**"Alternative NNUE inputs" (Oct 2025)** — cited on chessprogramming.org but
unretrievable during the survey (503s); it should be read before any
publication-novelty claim.

---

## 4. Prior art — ML

The survey covered six families; what follows is the distilled map from
method to our constraint. Full framing point first: the accumulator is a
**linear sketch over a multiset of atoms**, so the constraint is an *exact
homomorphism* requirement (turnstile streaming model), not approximate
locality. Methods split cleanly into those with the property by construction
(and no learning of the code) and those that learn codes without caring about
update cost.

### 4.1 What breaks

- **VQ-VAE** (van den Oord et al., NeurIPS 2017, arXiv:1711.00937): global
  encoder → one move rewrites arbitrarily many codes. Breaks F5.
- **Residual VQ / RQ-VAE semantic IDs** (SoundStream, arXiv:2107.03312;
  Rajput et al., NeurIPS 2023, arXiv:2305.05065): sequential stage dependence
  makes perturbation cascade — worst of the family.
- **Global top-k / k-WTA / k-sparse AEs** (Makhzani & Frey arXiv:1312.5663;
  Gao et al. arXiv:2406.04093): anti-incremental (see F5). Per-atom top-k is
  fine.
- **Sparse MoE routing** (Shazeer et al. ICLR 2017; Switch, JMLR 2022):
  routing on the whole input breaks; routing on atom identity reduces to
  hashing (below).
- **Learned index structures** (Kraska et al., SIGMOD 2018): optimizes lookup
  under a key distribution, not code stability — a dead end here.

### 4.2 What transfers

- **Feature hashing** (Weinberger et al., ICML 2009, arXiv:0902.2206):
  hashing a categorical ID space into B buckets satisfies every constraint
  *exactly* — the hash is atom-local by construction. Theoretical
  justification for shrinking the 60k threat vocabulary. Not learned.
- **Compositional/QR embeddings** (Shi et al., KDD 2020, arXiv:1909.02107):
  complementary partitions of an ID space (quotient + remainder), 15× table
  compression at negligible cost — **structurally identical to NNUE feature
  factorization** (virtual features), independently rediscovered by the
  recsys world, with six years of work on choosing *good* partitions.
- **Hash embeddings** (Svenstrup et al., NeurIPS 2017, arXiv:1709.03933):
  each token fires k rows from a shared pool via k hashes and **learns
  per-token importance weights over them** — the closest existing formalism
  to "learn which sparse rows fire per atom."
- **ROBE** (Desai et al., MLSys 2022, arXiv:2108.02191): block-contiguous
  shared arrays, 1000× compression, *faster* from cache locality — the
  engineering shape of a compressed threat table.
- **Unified Embedding** (Coleman et al., NeurIPS 2023, arXiv:2305.12102):
  one table serving many vocabularies — the analogue of sharing rows across
  our two channels.
- **Learned sketches** (Hsu et al., ICLR 2019): learn which items are heavy,
  give them dedicated buckets, hash the tail — directly the census-driven
  hybrid in §6.
- **L0 / hard-concrete gates** (Louizos et al., ICLR 2018, arXiv:1712.01312),
  **Gumbel-softmax/ST** (Jang et al.; Maddison et al., ICLR 2017),
  **DSelect-k** (Hazimeh et al., NeurIPS 2021, arXiv:2106.03760 — learnable
  binary codes selecting k of N with cardinality control): the training-time
  tools for differentiable selection over a discrete vocabulary.
- **AutoFIS/AutoCross** (Liu et al., KDD 2020, arXiv:2003.11235; Luo et al.,
  KDD 2019): relax the discrete choice of which feature *conjunctions* to
  instantiate into gated continuous parameters, then retrain on survivors.
  A chess feature set IS a set of conjunctions of atoms (king × piece;
  attacker × victim) — this is the pragmatic landing zone for vocabulary
  learning, and every candidate conjunction is atom-local, so incrementality
  is preserved *by construction*, not regularised for.
- **FSQ / lookup-free quantisation** (Mentzer et al., ICLR 2024,
  arXiv:2309.15505; MAGVIT-v2, arXiv:2310.05737): per-atom scalar/sign
  quantisation with no codebook — the most transferable quantiser: a per-atom
  encoder with an FSQ head learns a genuinely novel code and compiles to a
  table at export (DHE, Kang et al. KDD 2021, arXiv:2010.10784, shows the
  encoder-computes-embedding pattern).
- **Fly hashing** (Dasgupta, Stevens, Navlakha, Science 358:793, 2017):
  sparse random expansion + per-atom WTA — the **free random baseline any
  learned code must beat**, adopted as a control arm in §5.
- **SDR/VSA capacity theory** (Ahmad & Hawkins arXiv:1503.07469; Kanerva
  2009): the arithmetic for false-match probability and union capacity of
  w-of-n additive binary codes — the right formulas for sizing R and k.

### 4.3 Is our constraint a studied ML problem? (the publishable angle)

**No.** The survey found no paper that trains a code with an explicit
update-cost objective, and none that formalises the O(1)-delta requirement as
a learning constraint. The nearest neighbours, in order:

1. **Incrementally-Computable Neural Networks** (Sharir & Anandkumar,
   arXiv:2307.14988, 2023) — nearest in *goal*: quantise intermediates so
   edit cost scales with the modified fraction (12.1× median op reduction
   under atomic edits); but post-hoc, no guarantee, does not learn the index.
2. **The linear-sketch/hashing lineage** (Count-Sketch 2002; feature hashing
   2009; QR 2020; ROBE 2022; hash embeddings 2017) — nearest in *mechanism*
   (exact turnstile updates, additive over atoms); but the code is random,
   only values are learned.
3. **MONGOOSE** (Chen et al., ICLR 2021) and **learned locality-sensitive
   bucketing** (Chen & Shao, 2023) — nearest in *learning a stable hash*
   with a re-hash budget; but stability is statistical and codes are
   recomputed, not delta-maintained.

The correct related-work claim for a paper: *"chess NNUE is a deployed,
extreme instance of learned turnstile sketching — the code must be an exact
homomorphism over a structured atom set — and the ML literature has the
pieces (hash embeddings, gated conjunction search, per-atom quantisers) but
has never assembled them under an update-cost constraint."* The engine-side
survey (§3) supports the complementary claim that no engine has learned its
index function either.

Also noted for honesty: the nnue-pytorch docs state the constraint explicitly
("the inputs should change as little as possible between subsequent
evaluations") and record that "more sophisticated feature sets … usually
cannot combat the hit on performance" — the community's prior is that
*bigger handcrafted* sets fail on cost; *learned* sets at equal cost are
untried. The single academic NNUE feature-set study found (Martín Lombardo,
MSc thesis, U. Buenos Aires, Nov 2024, github.com/mlomb/cs-master-thesis)
evaluates only handcrafted sets (PDF not machine-readable at survey time —
flagged unverified).

---

## 5. The cheap pre-screen (built first, because everything routes through it)

A full retrain is ~14 h/arm; the screen must rank codes without it. Three
tiers, each with its null. The key enabling facts: (i) the trainer computes
feature indices in numpy (`jax/archon_jax/features.py`, index-exact against
the Rust sources), so **any candidate code is a remap of the index stream at
load time — zero trainer surgery**; (ii) the heldout-loss instrument
(`jax/tools/stage3_train.py`, `data/nnue/heldout/uho-band0-5.bin`) is the
project-proven ±4-Elo-resolution trend instrument; (iii) NPS does not depend
on weight *values*, so the speed question decouples from the quality question
and can be benched with a projected (untrained) net.

### 5.1 Tier 0 — minutes, no training, runnable beside a live GPU job

**T0a — the census (BUILT AND RUN, 2026-07-30).**
`tools/nnue/feature_census.py`: streams shuffled bulletformat records through
the trainer's own mapper and bincounts active indices. 4M positions of
`uho-band75-100.shuf.bin`, ~3 min nice'd CPU:

```
halfka: mean active/pos = 14.42      threat: mean active/pos = 12.36
halfka: rows ever fired 20310/22528 (90.2%)
  top-  1024 rows carry 68.85%   top- 4096: 89.82%   top- 8192: 97.02%
threat: rows ever fired 47378/60720 (78.0%)
  top-  1024 rows carry 56.38%   top- 4096: 80.55%   top- 8192: 90.41%
  top- 16384 rows carry 96.76%   99.9% of mass in top-38307
```

(The 14.4 + 12.4 ≈ 27 active rows/position cross-checks the known ~26.5-29.6
rows/perspective recipe density.) The skew is the empirical license for the
whole program: **1.7% of threat rows carry 56% of all threat-feature fire,
and 22% of rows never fire in 4M positions** — capacity allocated by hand to
events the data never sees. To run per-corpus (this is one band; the mix
matters) and to extend with an **importance** census: frequency ×
‖W_incumbent[row]‖ (rare-but-heavy rows are exactly the ones a
frequency-only merge would wrongly destroy — the blunder-tail guard, §7).

**T0b — the projection screen (closed-form, no training).** For a candidate
partition g of rows, the best tied table is the frequency-weighted mean
`W̄[g] = Σ f_t W[t] / Σ f_t`. Evaluate the *projected incumbent* (tied rows,
no retraining) on the heldout set and on a tactical FEN suite (Δeval
distribution, SF frame at the tool boundary). Minutes on GPU or CPU. This
bounds the *reconstruction* cost of a merge before any training; candidate
partitions that already project near-zero loss are near-certain survivors of
Tier 1.

**T0c — the cost counters (no training).** For context/vocabulary candidates:
replay recorded games / bench traces and count (a) context-transition rates
(the refresh dial, F2), (b) event counts per move under a modified `MAP`
(the filter dial, F3), (c) Finny staleness (set-diff sizes) under candidate
bucket tables. Pure counting; calibrate against the measured 0.1610 / 0.0214
/ 0.4151 constants. Output: amortized cycles via the §1 cost table, converted
to Elo at ~120 Elo/doubling.

### 5.2 Tier 1 — hours: the project-and-finetune panel

The flagship screen. Warm-start the full-width net from the incumbent with
tied rows initialized at the T0b projection, finetune ~10-20 superbatches
(~33-66 min/arm at D48X speeds), heldout at every save. The panel for the
first question ("is the threat code compressible and plastic?"):

| arm | what it tests |
|---|---|
| identity remap | the instrument null — must reproduce baseline exactly (also run twice to measure run noise) |
| random-partition, R'=8192 | the inert-code null (NULL FIRST): the damage a *content-free* code change does |
| fly-hash 2-of-8192 (random sparse expansion) | the no-learning random-code control the ML literature says is hard to beat |
| freq-k-means of incumbent rows, R'=8192 | the learned compression candidate |
| freq-k-means, R'=16384 | the dose-response point |

**Pre-registered decision rule.** Let Δ(arm) = heldout(arm) − heldout(identity)
at matched superbatch. (1) If Δ(random-8k) ≈ 0 within run noise, the
instrument cannot see even a violent code change — change instrument (per the
standing null-first rule; not expected, and it doubles as the noise floor).
(2) The learned code must beat the random partition by ≥3× run noise AND land
within the heldout delta historically corresponding to ≲10 fixed-node Elo.
(3) Learned-8k passing ⇒ the code is compressible 7.4× ⇒ proceed to the
retrain + the engine remap prototype. Learned-8k failing while learned-16k
passes ⇒ compressibility is real but shallower — still a result. All arms
failing (1)-(2) ⇒ the incumbent's code is not recoverable by clustering at
these R' — report the number, move to the splits/vocabulary arms (§6.3),
which do not depend on compression being true.

Cold-start control: one reduced-width (256) from-scratch micro-train on the
best candidate vs the identity code, ~1 h, to separate "the code is good" from
"the incumbent's weights survived projection" (the incest guard, §7 item 4).

**The parallel engine track (independent of training):** implement deploy
mode (ii) (§6.2), load the *projected* net, and bench NPS. The cache-speed
answer arrives without waiting for any training verdict.

### 5.3 Tier 2 — the one retrain, then games

Only the surviving candidate gets the ~14 h full retrain, and only a full
retrain's net gets games (the standing rule: heldout ranks, **games
arbitrate**, and the match launches the moment the net exists). Ledger rows
(D38) for the remap artifact (it is part of the net's identity — a net is now
(weights, code), and the code's sha256 belongs in the provenance row).

**Timeline from a standing start:** hour 0-1 Tier 0 (census done in 3 min;
projection + counters ~1 h); hours 1-6 the five-arm Tier-1 panel (GPU slack
required — must queue behind D49); day 2 cold control + decision + retrain
launch; day 3 games. **You know whether the representation is working at the
Tier-1 readout — hours of compute, not days.**

### 5.4 Known weaknesses of this screen, named

- Heldout loss ≠ Elo (the D48X promotion is the standing counterexample);
  the screen *ranks*, games decide.
- Project-and-finetune inherits the incumbent's distinctions — it screens
  *compression* perfectly and *discovery* not at all; discovery arms (§6.3)
  use the cold control and gradient-dispersion screens instead.
- One corpus band ≠ the mix; census and projection must be re-run on the
  training mix before any decision.
- Micro-width cold controls may not transfer to full width; they are a
  learnability check, not a strength ranking.

---

## 6. The concrete proposal — the Recode program

Staged so the cheapest, least-risky arm carries the first result, and each
stage's machinery is the next stage's instrument. All training-side; the
engine sees table contents until the final optional step.

### 6.1 R1 — Recode-T: a learned re-coding of the FullThreats channel

**What.** Replace the injective 60,720-row threat code with a learned
surjection onto R' ∈ {8192, 16384} rows, learned by importance-weighted
k-means over the incumbent's threat rows (T0a/T0b), refined by
project-and-finetune (§5.2), optionally polished by R3's re-assignment loop.

**Why threats first.** (a) The census shows the skew is dramatic there;
(b) the channel is the expensive one (77.7 MB table, #2 fold kernel, the
measured 1.25-1.39× speed cost of the whole channel); (c) its context is
1-bit, so re-coding cannot interact with refresh behaviour; (d) the row
space is pure relation-geometry — exactly the space where hand allocation is
least likely to match the data, and where SF's own history (§3.1) shows
families shipping dead. The win, if any, arrives in the currency SF's
history says to expect: **rows → width or speed** (both hm and threats paid
for L1 changes in the same commit); a compressed threat code that is merely
strength-neutral still converts to NPS via mode (ii), the same
quality-for-speed exchange PlentyChess measured (+45.5 fixed-node, −3.1
STC) from the opposite direction.

**Trainer changes** (all small): `features.py` applies `idx = remap[idx]`
before padding (identity = None); `spec.py` gains `threat_remap` (path) and
the arch's `l0_threatw` shape becomes (R', hl); `export.py` mode (i) scatters
tied rows back to full size. The Pallas embedding-bag kernel is untouched —
it consumes whatever indices the mapper emits.

**Two deploy modes — the staging trick:**
- **Mode (i), tied full-size table: ZERO engine change.** The CENN file still
  carries 60,720 rows; R' of them are distinct. Bit-level engine parity makes
  this the clean strength A/B (same binary, same gate machinery; re-pin the
  NNUE behavioural gate for the new net as usual).
- **Mode (ii), the real remap:** a `[u16; 60720]` table (121 KB, L2-resident)
  in a new CENN section; `ThreatResolver::index_decoded` returns
  `Some(remap[idx])` — **one line at the single chokepoint both the
  from-scratch and incremental paths already share**, so the two cannot
  drift; the weight table shrinks to R' × ACC (10.5 MB at 8k). ~30 lines in
  `sota_features.rs` + loader section + `net_scale_check` + gate re-pin.
  This is the arm with the cache-speed upside, benchable with a projected
  net before any training finishes.

### 6.2 R2a — the learned king-context coarsening (HalfKA)

Corrected by the §3.1 survey: at B = 32 with the mirror, the table is full
resolution — there is no alternative 32-partition to learn. The learnable
object is the **coarsening**: which of the 32 mirrored king squares share a
context, at B ∈ {8, 12, 16} — the region where the rest of the field lives
(Reckless 10, Integral 12, Berserk 16) with layouts chosen entirely by hand
and **no published attempt to learn one, anywhere**. Learning it: cluster
per-king-square weight structure — either the incumbent's per-bucket row
blocks directly (they are per-king-square already, since B = 32 is full-res:
a pure T0b-style projection + k-means arm), or a factorized correction table
trained briefly on a frozen trunk if finer statistics are needed. The value
proposition is data density + a 28.8 → ~10 MB table (the same cache argument
as §6.1) at some specificity cost; Reckless's 8-of-10-buckets-on-ranks-1-2
layout is the hand prior a learned layout must beat. Screens: T0c staleness
counting per candidate B, projection + finetune as in §5.2. Engine change:
the const table becomes net-carried (64 bytes); `halfka_king_key`/
`halfka_row_base` read it; Finny sizing follows B. The honest prior: our
training corpora are large enough that full resolution may simply be right
(the folklore threshold is ~1B positions and we exceed it) — but the arm is
nearly free, and a confirmed null pins the "hand full-res was already
optimal at our data scale" boundary, which the paper needs anyway.

### 6.3 R2b — vocabulary arms (the `MAP` mask)

- **Drop arms (speed):** rank relation families by importance mass (T0a ×
  ‖W‖); zero candidate families in the incumbent and measure Δeval (no
  training), then micro-train without them. A family whose removal is
  eval-neutral pays pure event-count savings (T0c), i.e. NPS. Engine change:
  `MAP`/`KEPT_VICTIMS` consts (single source of truth, compile-time derived).
- **Add arms (strength) — with the priors the survey set.** King-victim
  relations looked like the natural candidate (threats *on the king* ≈
  check/king-pressure structure), but §3.1 sets a **negative prior**: they
  were in SF's original 79,856-dim set and were removed in SFNNv11 as a
  non-regression. Re-adding them in ARCHON is a cheap arm (mask flip +
  zero-init rows + warm finetune) but should be ranked below families with
  *positive* evidence: SFNNv14's pawn-push rows and SFNNv16's PP_3Wide
  pawn-pair family both passed as gainers **after** the walk-down — the
  vocabulary is demonstrably still in motion, and pawn-structure relations
  are its live frontier. (PP_3Wide is a genuinely new atom family — pawn
  pairs — proving the observer class extends beyond attack relations.)
  Either way the shape of the experiment is the same: T0c event counting
  first, then the §5.2 screen; the *learned* version of this arm is R3's
  gates ranging over a candidate superset rather than one family at a time —
  SF's one-family-per-PR walk-down, done wholesale.

### 6.4 R3 — learning the code end-to-end (the research flagship, gated on R1)

If R1 shows the code is plastic (learned-8k ≫ random-8k), graduate from
clustering-the-incumbent to learning the assignment during training:
alternating optimization in the VQ-EMA style — train the tied table; every N
steps re-assign tuples to rows by nearest-centroid in weighted
gradient/feature space, EMA centroids, dead-row re-init, assignment-churn
caps (the D47M unopposed-penalty lesson governs any gate/temperature
annealing: ramp, never step, when a resisting gradient source is removed).
The differentiable alternatives (hard-concrete L0 gates per tuple-row pair;
DSelect-k per tuple) are the literature-standard forms if the alternating
loop oscillates. Splits are proposed by per-tuple gradient dispersion within
a row (computable on sampled batches with the frozen trunk), merges by
centroid distance — a split-merge local search, each move validated by the
§5.2 screen. **This stage is the paper**: a learned turnstile code under an
explicit update-cost budget, with fly-hash and random-partition controls and
a deployed-engine evaluation.

### 6.5 Is `ft_sparsity_lambda` the right knob? (the coordinator's question)

**No — different tensor, different objective.** Verified in
`jax/archon_jax/train.py` (L1 on the post-CReLU *accumulator activations*,
driving deploy *density* for the `find_nnz` path) and ruled to default 0 on
2026-07-26 (mis-shaped vs the engine's per-4-group cost; λ=0.001 never even
moved density). Nothing in this study penalizes activations. What *is*
directly reusable from that machinery: the warmup-ramp pattern
(`ft_sparsity_warmup_superbatches` + the D47M lesson) for annealing R3's
gates, the loss-plumbing seam it demonstrates, and — more importantly — the
loader remap seam and heldout instrument named in §5. If R3 adopts
hard-concrete gates, the penalty enters as a *new* term on gate
probabilities (an L0 on the assignment, not an L1 on activations), with the
existing ramp field as its schedule template.

---

## 7. Honest assessment — what would make this fail

1. **The hand code is already near-optimal for strength.** SF's feature sets
   carry years of fishtest selection, and the nnue-pytorch docs record that
   richer sets historically die on cost. Compression may then be
   strength-neutral — but the program still returns the mode-(ii) speed win
   and the science result, and the 22%-never-fired census row says capacity
   *allocation* has headroom even if the distinctions are right.
2. **The blunder tail.** Rare threat rows may encode exactly the sharp
   tactical relations whose merging creates the catastrophic-loss class the
   project has ruled unacceptable. Guard: importance-weighted (not
   frequency-only) clustering; the tactical-FEN Δeval screen in T0b; a
   per-class readout (never a lone median) at every screen.
3. **Heldout ≠ Elo, and the era is confounded.** The current canonical net
   was promoted while *losing* games on the ruling that the search cannot yet
   exploit it — any strength read on an eval-side change is muddied right
   now. Games at Tier 2 arbitrate; expect wide bars; quote levels, not
   trends.
4. **Incest in the screen.** Codes derived from and judged against the
   incumbent cannot discover better distinctions — only cheaper ones. The
   cold-start control and the R2/R3 discovery arms carry that burden; if
   only the compression arms ever pass, say so — that is a real (and useful)
   negative about this route to discovery, not about compression.
5. **k-hot economics.** The F4 arithmetic likely kills runtime factor-row
   proposals at current NPS; if eval quality per row turns out to need k>1,
   the program stalls at k=1 results. The fold-cost model itself is an
   estimate from within-run attribution shares — re-measure before trusting
   any borderline verdict.
6. **Tail estimability (R3).** Per-tuple gradient statistics at the rare tail
   — where code choice matters most — are noisy where data is thinnest; the
   re-assignment loop may thrash there. Churn caps and heldout-gated moves
   mitigate; they do not remove the risk.
7. **Scheduling.** Tier 1 needs GPU slack behind the live D49 run; the only
   parts runnable today are Tier 0 (done/CPU) and the engine remap prototype.

## 8. What I am least sure about

1. **Whether the mode-(ii) cache win is real.** The miss profile may be
   dominated by a hot row subset that is already resident, in which case
   shrinking the cold tail buys little. It is also the cheapest uncertainty
   to kill: the remap prototype + a projected net + one bench session.
2. **Whether heldout-ranked code quality transfers to Elo in the current
   regime** (see §7.3). This uncertainty sits under every screen in §5, and
   only Tier-2 games resolve it.
3. **Whether any learned 32-way king partition can beat SF's hand table.**
   My prior is weakly pessimistic (it is the most-tested constant in the
   feature set) — but the arm is nearly free and a confirmed null pins the
   "hand-design was already optimal *here*" boundary, which the paper needs
   anyway.
4. **The clustering metric for R1.** Frequency-weighted L2 on incumbent rows
   is the obvious choice and may be the wrong one (rows interact through the
   shared nonlinear head; two rows can be distant in L2 and interchangeable
   in effect). If Tier-1 verdicts look noisy, the metric is the first
   suspect — the T0b projection screen can compare metrics cheaply.
5. **Two unverified prior-art items.** The Lombardo thesis PDF (the only
   academic NNUE feature-set study found) resisted machine reading, and the
   talkchess "Alternative NNUE inputs" thread (Oct 2025) could not be
   retrieved during the survey. Both must be read manually before any
   publication-novelty claim; either could contain a prior attempt.
