# The representation gap — what the trained network provably cannot read, measured (2026-07-30)

**Status:** IN FLIGHT — this header is finalized in the same session once the
probe batteries and value arms land. Branch `repr-gap-study`. Predecessors
(all read; none continued): `recode-poc-2026-07-31.md` (compression — cost),
`richer-representation-study-2026-07-31.md` (refinement pricing — cost, one
7× correction), `learned-representation-study-2026-07-30.md` (update-locality
— cost), `conjunction-discovery-study-2026-07-30.md` on `conj-study`
(learned selection over existing atoms — capacity, not new information).

**The commissioned question:** build a representation whose content is
DISCOVERED BY TRAINING over something the current network has NO ACCESS TO
AT ALL — not a recombination/compression/selection of existing atoms, not a
hand-picked feature family.

---

## 1. Theory: three levels of "no access", and why presence is not access

HalfKA+threats is injective on placement (pinned:
`threats_active_in_range_and_distinct`, sota_features.rs:1108), so every
static fact is *present* in the input in the information-theoretic sense.
Access is a different property. The architecture is

    acc(x) = b + Σ_{f: x_f=1} W_f                       (linear, 83,248 -> 1280×2)
    h_i    = crelu(acc_i)·crelu(acc_{i+640})            (640 products/persp)
    + two tiny dense layers (L2 16, L3 32, 8 buckets) + psqt/skip lanes

and the *accessible* information at the readout is what decoders of this
class can extract from `acc`. That yields three distinct levels of "no
access at all", each with its own measurement:

- **L1 — function-class blindness.** Deterministic functions of placement
  (exchange outcomes, race outcomes, parity facts) whose computation needs
  more sequential depth than one linear map + one product + 48 dense units.
  The information is in the input; the function class cannot surface it.
  *Measurement: capacity-stratified decodability probes on the frozen acc.*
- **L2 — feature blindness.** State dimensions never encoded: **castling
  rights, en passant, rule50, repetition/path do not appear anywhere in
  sota_features.rs** (verified by grep this session, whole file). rule50
  touches the EVAL only as the hand-designed outer damping
  `v -= v·rule50/RULE50_DAMPING_DEN()` (src/eval/mod.rs:574) — position-blind,
  linear, outside the net; castling/ep/repetition touch nothing.
- **L3 — data blindness.** The training records themselves erase the
  dimension: bulletformat's 32-byte record (spec.py RECORD_DTYPE) carries
  occ/pieces/score/result/ksq/opp_ksq/move — **no castling, no ep, no
  rule50** (data.py:317 documents the drop). The labels were produced by
  engines that SAW the full state, so for every rights/clock-sensitive
  position the label contains variance the input provably cannot explain —
  irreducible label noise the incumbent absorbs as a mixture average, at
  every one of its ~7B training-position visits.

Castling/ep/rule50 sit at L2+L3 simultaneously — the cleanest possible
reading of "NO ACCESS AT ALL": the net doesn't see them, and could not have
learned them from its data if it did.

## 2. Method — three instruments, each with a null, closing the loop from "missing" to "worth supplying"

The commissioned method (probe the frozen accumulator for decodability) is
instrument 1. Two additions make it decisive rather than descriptive; this
is the claimed improvement on the brief:

1. **Decodability probes, capacity-stratified.** The conj study proved the
   sum is generically injective on sparse sets, so an UNBOUNDED probe could
   decode anything and "not decodable" would be meaningless. Probes are
   therefore a ladder pinned to the architecture's own readout classes:
   P1 linear on acc (2560); P2 linear on [acc, h] — h the exact 640-product
   features, i.e. the function class of the out_skip lane and everything the
   present head reads linearly; P3 ridge on 3072 frozen random ReLU features
   of [acc,h] — a shallow-MLP-class upper bound, closed-form trainable.
   P0 = ridge on the net's own output logit (what the net already says).
   Controls: material/pcount MUST decode; an occ-hash bit MUST NOT; every
   target also probed against shuffled labels (the null column).
2. **Value probes on the same frozen acc.** For each candidate quantity T:
   ridge [rf(acc,h)] → training label vs [rf(acc,h), T] vs [rf(acc,h),
   shuffled-T], trained on one set, MSE read on another, in the exact panel
   target semantics (wdl .3, sigmoid s/400 — heldout deltas are in the
   Tier-1 panels' units, exchange rate 12–28e-6/Elo). This is the learned
   embedding in miniature: WE name only the information source; the probe
   learns its use. A quantity that fails decode AND moves the label MSE is a
   proven hole with proven value — that composition, not either number
   alone, is the deliverable.
3. **Oracle twins (planned arm, §7).** For L2/L3 state: pairs identical to
   the net, differing only in the invisible dimension, scored by SF
   eca43a97 at fixed nodes — a value ceiling independent of our net and of
   probe capacity.

Everything runs CPU-side against the frozen canonical D48X (sb520-EMA raw,
the panels' warm point); the GPU stays the D49 trainer's.

### 2.1 Validity gates, all PASSED before any probe ran

- **Acc extraction parity:** recomposed logit (head_forward(acc)+psqt) vs
  model.forward, first 3 batches of every run: max|Δ| ≤ 1.2e-6 (f32
  jit-order noise; gate 1e-5). Same graph, same weights.
- **Encoder roundtrip:** full-state FEN → canonical record reproduces the
  real pipeline's bytes on 20,000/20,000 heldout records
  (occ/pcs/ksq/opp_ksq all byte-identical — pins the nibble order and the
  enemy-frame opp_ksq convention). Flip involution 2000/2000.
- **SEE selftest:** classic CPW positions (defended/undefended/x-ray
  battery) pass sign and value.
- **Windows:** probe TRAIN = uho-band75-100.shuf.bin rows [5.0M, 5.35M) —
  disjoint from every prior study's windows (census [0,2M), conj probe
  [2M,3M), confirm [3M,3.5M), region probe [4M,4.51M)). Probe EVAL =
  uho-band0-5.bin (the panels' heldout file). Full-state sets: 450k
  (filtered TSV) + 250k (raw TSV), fresh shuf samples, seeds 42/43.

## 3. The oracle-twin results (L2+L3) — the invisible state is decision-relevant

Twin pairs are IDENTICAL to the network and differ only in the invisible
dimension; SF eca43a97 (the pinned constants reference) at 60k fixed nodes
arbitrates; ARCHON receipts prove the blindness end-to-end. n=800 each, from
the filtered (trainer-parity) full-state corpus. Discrete outcome first.

| twin | ARCHON static eval | SF bestmove changes | SF |Δcp| p50/p90/p99 | mean Δ |
|---|---|---|---|---|
| castling rights (stm owns) vs stripped | **800/800 bit-identical** | **312/800 = 39.0%** | 18 / 78 / 167 | +20.0 cp to the owner |
| rule50 = 0 vs 90 (no rights/ep) | **800/800 bit-identical** | **299/800 = 37.4%** | 11 / 57 / 247 | −2.8 cp |

Readings, kept one inference deep:

- **In ~4 of 10 rights-bearing positions the correct MOVE depends on
  information our evaluation provably cannot see.** The engine's search
  still knows the rights (movegen), but every leaf eval scores
  rights-sensitive positions identically — the tree is steered by an eval
  that cannot tell a castled future from a lost one.
- The castle Δ is one-signed (+20 mean: rights help their owner), so the
  net's mixture-average has learned "some rights-shaped placement bonus"
  smeared over both cases; the per-position error is the p50–p99 column.
- rule50's mean ≈ 0 but |Δ| tail is FAT (p99 247 cp): the clock converts
  advantages to draws symmetrically. The NET is fully clock-blind (800/800);
  the engine's only defense is the outer damping — position-blind and
  linear, i.e. exactly the mixture-mean of the interaction the oracle shows.
- Caveats: SF's twin Δ includes search-strategic effects (e.g. rushing
  progress near the clock), not pure eval differences; 60k nodes is the
  budget the box allows beside the trainer. Both twins were drawn from
  rights/clock-UNfiltered rows of the corpus, so prevalences below scale
  the per-position numbers into per-game terms.

### 3.1 Prevalence census (the L3 exposure)

Full-state UHO corpus, 450k filtered / 250k raw samples:

| state | filtered | raw |
|---|---|---|
| any castling right | 2.7% | 3.1% |
| stm owns a right | 1.4% | 1.8% |
| ep square set | 0.06% | 0.11% |
| rule50 ≥ 20 | 11.2% | 9.6% |
| rule50 ≥ 40 | 5.6% | 4.7% |
| rule50 ≥ 80 | 1.1% | 1.0% |

Two honest readings. (a) The corpus under-represents rights positions —
UHO book exits are deep, both sides have usually castled — so the net gets
FEW examples of exactly the positions where the twin gap is largest: the
label-noise problem and the data-scarcity problem stack. (b) Deployment
value scales with IN-GAME prevalence (every game passes through a
rights-bearing opening), not corpus prevalence; an in-tree rights-prevalence
counter (foldcount pattern) is the cheap follow-up that converts §3's
per-position numbers into per-game terms. The rule50 tail needs no such
argument: ≥40 is 5.6% of the training distribution itself.

## 4. Decodability results (L1)

RESULTS SLOT — the decode table: target × {P0, P1, P2, P3, P3-null},
controls first, with n and base rates.

## 5. Value results

RESULTS SLOT — label-MSE deltas per supplied quantity vs shuffled control,
in panel units; the netout reference row (must reproduce ~0.0064 on the
panel heldout).

## 6. What this study does NOT do

- No hand-designed feature ships from here: any deficit that survives goes
  into the trainer through a raw stream + learned embedding (the seam is
  the same map_batch/arch machinery every prior panel used), with a
  content-free control arm, pre-registered before launch.
- Costs are used to REJECT candidate streams only (−5.38 Elo per extra
  threat row/node, −10.38 per HalfKA row; TT-cacheability: any
  path-dependent stream is flagged, not smuggled).

## 7. Planned arms beyond v1 (in priority order)

1. Oracle twins for castling/rule50 (SF eca43a97 fixed-node, discrete
   outcomes preferred: bestmove-change rate + Δcp distribution) + the
   ARCHON blindness proof (nnue-eval on twins must be bit-identical for
   castling).
2. KPK exact-result probe (synthesized, retrograde-solved — the exact
   game-theoretic target for the deepest function-class question).
   Pawn-endgame subset targets (opposition, square rule) measured at 0.1–0.2%
   prevalence on the big sets — deferred to a dedicated endgame extraction.
3. The in-architecture learned-embedding panel for the winning deficit
   (state atoms / exchange stream through the map_batch seam; learned vs
   shuffled-control arms; 12-sb finetune recipe; queued behind the GPU
   queue on the owner's say-so).
