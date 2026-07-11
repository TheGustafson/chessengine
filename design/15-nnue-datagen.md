# 15 — NNUE training-data generation (bootstrap datagen + curation)

Status: Draft v1

This document specifies how we produce the training corpus for the Phase C
NNUE net (D21/D22, doc 11 §3.4): the exact on-disk record format the
`bullet` trainer consumes, the curation/filtering rules that decide which
positions enter the corpus, the teacher-labeling settings, the volume and
storage plan, and the D28 two-tier provenance progression (distill from a
locally-compiled Stockfish to *de-risk the NNUE infrastructure*, then
retrain the shipped net from clean self-play data). It exists because
getting curation wrong wastes the bulk labeling compute: this is the
"think hard about the corpus before spending the core-hours" contract.

The bootstrap corpus and its Stockfish teacher are **discardable
development scaffold** (D28): the SF-labeled data and the compiled SF
binary are build artifacts, gitignored, never shipped; the released net is
trained only on our own/self-play data (D27).

---

## 1. Purpose and scope

**This doc owns:**

- The **training-data format contract**: the exact `bullet` /
  `bulletformat` record the corpus must be in (§3.2), the text interchange
  line our generators emit, and the score/result perspective convention
  (the sign-flip trap, §3.2.4). *Every* data generator we write — the
  bootstrap SF tool here and the engine's own clean self-play generator
  (doc 11 §3.4.7, `bin/datagen.rs`) — emits this format and obeys this
  curation spec.
- The **curation / filtering spec** (§3.5): position selection, quiet /
  non-noisy filtering, in-check exclusion, score capping, deduplication,
  phase / piece-count balancing, book-ply skipping.
- The **teacher-labeling settings** (§3.3): the locally-compiled Stockfish,
  its per-position search budget, and how its score maps to the corpus.
- The **bootstrap pipeline** (§3.4, §3.7): the SF-teacher orchestration
  tool (`tools/nnue/`), the game loop, adjudication, sharding, conversion,
  shuffle/interleave into a trainable corpus.
- The **volume + storage plan** (§3.6): target counts, the gitignored
  `data/nnue/` layout, regenerability metadata.
- The **D28 progression** (§3.1): bootstrap corpus → validate NNUE infra
  reaches strength → regenerate clean self-play corpus → retrain shipped
  net.

**This doc does NOT own** (it only defines the contract with):

- The **NNUE architecture, quantization scheme, and CENN net file format** —
  owned by doc 11 §3.4 (768→2×256→1, SCReLU, `QA=255`/`QB=64`/`SCALE=400`,
  the `CENN` container, `netpack`). This doc pins the *training* knobs that
  must match those constants, and stops at handing bullet a corpus + a
  recommended config.
- The engine's own **clean self-play generator** `bin/datagen.rs` (doc 11
  §3.4.7) — its implementation (PRNG reuse, per-worker TT, `MakeMoveHooks`)
  is doc 11's; it *conforms to* the format (§3.2) and curation (§3.5) this
  doc owns.
- The **texel-tuning corpus** and the `tuner` bin (doc 11 §3.3.3). The
  quiet-position filters here are shared with texel tuning by design, but
  the tuner's own EPD loader and public-dataset licensing question are doc
  11's (§9.2 there, now cross-referenced to D28 + this doc).
- **Make/unmake, Zobrist, movegen** (docs 04/05) — the bootstrap tool uses
  python-chess for rules, not the engine; the clean generator uses the
  engine.
- **SPRT/gauntlet strength gates** (D25, doc 14) — this doc names the gates
  the corpus must clear; doc 14 owns the methodology.

## 2. References

- `design/decisions.md` — **D28** (NNUE bootstrap two-tier strategy), **D27**
  (never copy SF code, never ship SF nets), **D21/D22** (eval roadmap, the
  `evaluate` seam), **D15** (external tooling ≠ dependency; `rand` banned in
  the crate).
- `design/11-evaluation.md` §3.4 — the NNUE architecture this data trains:
  §3.4.1 the 768 feature set, §3.4.3 quantization (`QA/QB/SCALE`), §3.4.5
  the CENN net format, §3.4.7 the training pipeline and the engine's own
  self-play generator. §9.2 dataset-licensing (now points here).
- `bullet` NNUE trainer — github.com/jw1912/bullet, `docs/1-basics.md`,
  `docs/3-data.md`: the `Chess768` input type, `DirectSequentialDataLoader`
  for flat `bulletformat` records, the binpack loaders (`SfBinpackLoader` /
  `MontyBinpackLoader` / `ViriBinpackLoader`), the `--wdl` (result-weight)
  and `--scale` (sigmoid) knobs, SCReLU activation. External tooling (D15).
- `bulletformat` — github.com/jw1912/bulletformat: the `ChessBoard` 32-byte
  record (§3.2.1 transcribes its layout) and the text interchange line
  `<FEN> | <score> | <result>` (white-relative). Facts about a data schema,
  reimplemented; no code copied (D27).
- `stockfish_reference/Stockfish/src/nnue/` — read for the SF feature set
  (`features/half_ka_v2_hm.h`: king-bucketed, mirrored, `11*64*64/2` dims,
  32 active features) and quantization provenance (`nnue_common.h`:
  `FtMaxVal=255`, `WeightScaleBits=6`, `OutputScale=16`). We deliberately
  start with the simpler unbucketed 768 (doc 11 §3.4.1); notes only, no code
  (D27).
- `python-chess` 1.11.2 (installed) — rules/FEN/adjudication oracle for the
  bootstrap tool.
- Existing strength harness: `tools/gauntlet.sh`, `tools/gen_openings.py`,
  the balanced opening book under `engines/` — reused for opening diversity.

## 3. Design

### 3.1 The two provenance tiers (D28)

The net is developed AlphaGo → AlphaZero style, in two strictly separated
data tiers:

```
 TIER A — BOOTSTRAP (discardable scaffold, this doc's main subject)
   locally-compiled Stockfish  ──teacher──►  data/nnue/ corpus (SF eval + WDL)
        │                                          │
        │                                    bullet train (768→2×256, SCReLU)
        ▼                                          ▼
   never shipped; deleted                    net-0 (CENN) ──SPRT──► vs Phase B HCE
                                                   │
                              GATE: net-0 clearly beats HCE  ⇒  NNUE INFRA PROVEN
                              (accumulators, loader, SIMD, trainer, format all work)

 TIER B — SHIP (clean provenance, D27)
   OUR engine self-play  ──own labels──►  clean corpus (bin/datagen.rs, doc 11)
        │                                          │
        │                                    bullet retrain (may be a full
        ▼                                    self-play "zero" restart)
   the released artifact  ◄──SPRT gated──   net-1+ (trained ONLY on our data)
```

The point of Tier A is to **decouple the machinery risk from the
data-quality risk**. Stockfish is a known-good teacher; if the pipeline
produces a net that beats our HCE, then the *infrastructure* — dirty-piece
accumulators, the CENN loader, the scalar/AVX2 backends, the bullet
training loop, and this data format — is proven. Only then do we spend the
larger effort on clean self-play data for the shipped net. SF-labeled data
and the SF binary are **build artifacts, never committed, never shipped**
(D27/D28).

### 3.2 What `bullet` consumes — the exact on-disk format

#### 3.2.1 `bulletformat::ChessBoard` — the 32-byte record (the target)

bullet's simplest, fastest-to-load trainable format is a flat array of
fixed 32-byte `bulletformat::ChessBoard` records, read by
`DirectSequentialDataLoader`. Each record is (byte offsets, little-endian):

| off | size | field       | meaning |
|-----|------|-------------|---------|
| 0   | 8    | `occ: u64`  | occupancy bitboard; set bit *b* ⇒ a piece on square *b*. Square order is little-endian rank-file (a1 = bit 0 … h8 = bit 63) — **identical to our D3 mapping**. |
| 8   | 16   | `pcs: [u8; 16]` | one **4-bit nibble per occupied square**, in ascending square order (the set bits of `occ`, LSB→MSB). Nibble *i* is the low nibble of byte `i/2` when `i` is even, high nibble when odd. Nibble value = `(colour << 3) | piece_type`, `colour` 0 = White / 1 = Black, `piece_type` 0=P,1=N,2=B,3=R,4=Q,5=K. |
| 24  | 2    | `score: i16` | evaluation in centipawns, **White-relative** (see §3.2.4). |
| 26  | 1    | `result: u8` | game outcome, White-relative: 0 = White loss, 1 = draw, 2 = White win (i.e. `2 × wdl_white`). |
| 27  | 1    | `ksq: u8`    | side-to-move king square (post-canonicalisation). |
| 28  | 1    | `opp_ksq: u8`| opponent king square, stored `^ 56`. |
| 29  | 3    | `extra: [u8;3]` | padding, zero. |

Total **32 bytes**, `#[repr(C)]`, compile-time size-asserted in
bulletformat. The `ksq`/`opp_ksq` fields are unused by a plain 768 net but
future-proof the *same corpus* for king-bucketed inputs (HalfKA, the doc 11
§3.4.1 upgrade path) — a reason to record the full board rather than a
768-only projection.

**We do not hand-pack this record.** bulletformat's constructor performs an
internal side-to-move canonicalisation (when Black is to move it vertically
flips every bitboard via `swap_bytes`, swaps the White/Black occupancies so
the mover is index 0, negates the score, and flips the result). Replicating
that bit-for-bit is exactly the kind of avoidable error a bootstrap should
not risk. Instead we emit the **text interchange line** (§3.2.2) and let
bulletformat's converter build the binary. The caller-facing contract of
both the text line and the constructor is **White-relative**; the canonical
STM storage is bulletformat's concern, not ours.

#### 3.2.2 Text interchange line + conversion pipeline

Our generators write plain-text shards, one position per line, in
bulletformat's documented chess text schema:

```
<FEN> | <score> | <result>
```

- `<FEN>` — standard FEN; encodes side-to-move, castling, ep (so the
  converter derives STM and the king squares itself).
- `<score>` — **White-relative** centipawns, integer (§3.2.4).
- `<result>` — **White-relative** game outcome: `1.0` White win, `0.5`
  draw, `0.0` White loss.

Pipeline to a trainable corpus:

```
generator ──► data/nnue/raw/*.txt              (text shards)
   bulletformat text→bin convert ──► data/nnue/bin/*.bin   (32-byte records)
   bullet-utils shuffle (per shard) ──► data/nnue/shuffled/*.bin
   bullet-utils interleave ──► data/nnue/train.bin  (+ held-out val.bin)
   bullet train (DirectSequentialDataLoader) ──► raw quantized net
   netpack (doc 11 §3.4.7) ──► CENN container
```

Text is a *transient* intermediate (~70–90 B/line): converted to the
32-byte binary and then deleted. The exact CLI of the convert / shuffle /
interleave utilities is pinned against the vendored bullet + bulletformat
commit at implementation time (§9, open question 1); their *contracts* —
text→bin, in-place shuffle, N-way interleave — are stable and are all this
doc depends on.

#### 3.2.3 Why `ChessBoard` / `Chess768` (not binpack, not HalfKA)

bullet can also load compact game-oriented "binpacks" (`SfBinpackLoader`,
`MontyBinpackLoader`, `ViriBinpackLoader`; Viriformat is popular for
self-generated data). Those store whole games + move sequences + scores and
support in-trainer rescoring, at the cost of game reconstruction and
format-specific packers. For the bootstrap we choose the **flat
`ChessBoard`** record because:

- It is the **exact match** for bullet's `Chess768` input type, which
  produces precisely the `2 × 6 × 64 = 768` perspective features doc 11
  §3.4.1 specifies (`idx = ((c != p) as usize * 6 + pt) * 64 + rel_sq(p,
  sq)`). No feature-set translation, no king buckets, no refresh logic.
- It is the simplest thing to emit correctly from either generator and to
  validate against python-chess (§5).
- The full-board + king-square record is a **superset** of what the 768 net
  needs, so the same corpus trains the doc 11 upgrade path (HalfKA,
  buckets) without regeneration.

Binpacks (and their rescoring) are revisited only if corpus *storage* or
*relabelling throughput* ever dominates (§9, open question 4). We do **not**
consume Stockfish's own binpacks or nets (D27).

#### 3.2.4 Score & result perspective — the sign-flip trap

The single most common datagen bug. The convention, end to end:

- **Stockfish reports `score cp` from the side to move.** So
  `score_white = (stm == White) ? sf_cp : -sf_cp`.
- **We write `score_white` (White-relative) into the text line**, matching
  bulletformat's "score is white relative" contract. bulletformat then
  re-canonicalises to STM-relative internally (§3.2.1).
- **Result is the game's White-relative outcome** (`1.0/0.5/0.0`) attached
  to every sampled position of that game — independent of whose move it is.
- At train time bullet applies `sigmoid(stored_score / SCALE)` with
  `SCALE = 400` (doc 11 §3.4.3) to build the eval target, and the STM-half
  ordering of the two accumulators (doc 11 §3.4.1) is what makes the trained
  net side-to-move relative. Our engine's `evaluate` returns STM-relative
  (doc 11 §3.1), so the two conventions meet: **train STM-relative, infer
  STM-relative**, mediated by White-relative on disk.

The doc 11 §3.2.6 / §3.4.2 symmetry test (`evaluate(pos) ==
evaluate(flipped(pos))`) is the runtime tripwire that this convention was
honoured through training and inference.

#### 3.2.5 WDL vs eval blend — the lambda convention (footgun)

bullet's loss blends the net's prediction toward *both* the eval and the
game result:

```
loss = wdl · MSE( sigmoid(pred/SCALE), result )
     + (1 − wdl) · MSE( sigmoid(pred/SCALE), sigmoid(score/SCALE) )
```

`--wdl` is the weight on the **game-result** term (`1.0` = pure WDL,
`0.0` = pure eval). Doc 11 §3.4.7 phrases the blend as "`lambda ≈ 0.7`
score / 0.3 result" — that is the *eval* weight. **They are complementary:**
doc 11's `lambda = 0.7` ⇒ bullet **`--wdl 0.3`**. We record *both* `score`
and `result` per position so the blend is a pure train-time knob; the
bootstrap default is `--wdl 0.3 --scale 400` (lean on SF's strong eval,
keep a WDL anchor). Later nets sweep `--wdl` under SPRT.

### 3.3 The teacher: locally-compiled Stockfish

#### 3.3.1 Build

The teacher is the Stockfish clone at
`stockfish_reference/Stockfish/` (already gitignored), built with the fast
profile (`make -j profile-build ARCH=x86-64-avx2`, or the repo's supported
`RUSTFLAGS` analogue for SF's `ARCH`). It links SF's own embedded net
(GPLv3) — which is exactly why its **output is scaffold**: the binary and
any data it labels are build artifacts, never committed, never shipped
(D27/D28). The packaged SF16 under `engines/` stays the strength-testing
*opponent* (doc 14); the teacher is a separate, stronger, locally-built
binary. Its git rev + `sha256` go in the corpus metadata (§3.6) for
regenerability.

#### 3.3.2 Labeling settings — fixed nodes, default 5000

Each move in a self-play game is chosen by SF under a **fixed node budget**,
and SF's reported `score cp` + `bestmove` for that position are the label.
One search yields *both* signals (eval + the move to play), the most
compute-efficient labeling.

- **Node-based, not depth-based** (`go nodes N`): nodes give near-constant
  work per position (depth's time variance is 10–100×), so throughput and
  the corpus's effective label strength are uniform across the game.
- **Default `N = 5000` nodes/move.** Justification: SF at 5000 nodes is
  still far stronger (~2500+ Elo equivalent) than our Phase B engine, so
  labels are high quality, while a 5000-node search is sub-millisecond —
  hundreds to low-thousands of accepted positions/sec/core (§6). This
  matches doc 11 §3.4.7's 5000-node budget for the clean generator, so the
  two tiers share a label-strength scale. `--nodes` is a CLI knob.
- **Optional high-node relabel pass** (deferred, §9 open question 3): take
  the *accepted* positions and re-search at e.g. 100k nodes for sharper
  eval targets. Higher quality, ~20× the per-position cost; not needed to
  prove infra, so off by default.

#### 3.3.3 Score scaling

We record SF's **UCI-normalised `score cp`** verbatim (what python-chess
exposes as `info["score"].pov(WHITE).score()`), White-relative. No manual
rescaling: bullet's `sigmoid(cp / 400)` fits the WDL model to whatever
consistent centipawn scale we feed, and `SCALE = 400` is doc 11's constant.
If the fitted `sigmoid(cp/400)` vs actual-result curve is visibly off on
the validation sample, `SCALE` is the single re-fit knob (§9 open
question 5). Mate scores are **not** rescaled — they are filtered out
(§3.5).

### 3.4 Position sourcing & the game loop

Fixed-node self-play games never terminate on their own, so every game ends
by an explicit rule and carries a well-defined White-relative result
`R ∈ {1.0, 0.5, 0.0}` applied to all its sampled positions.

**Opening diversity.** Each game starts from a varied opening, then plays
out under SF self-play:

- Default: play `R_open` random plies from the start position
  (`R_open` drawn uniformly in `[8, 10]` per game) choosing uniformly among
  legal moves, then a **balance gate**: reject-and-retry the opening if a
  quick SF eval of the resulting position exceeds `±600 cp` (avoids seeding
  games from already-lost positions, which skew WDL and waste labels).
- Alternative: sample lines from the balanced opening book already used by
  `tools/gauntlet.sh` (`tools/gen_openings.py` output) — better-distributed
  and pre-balanced. Selectable by CLI; the book path is preferred once a
  book is wired.
- The opening plies themselves are **not recorded** (they are off-teacher
  and near-book); recording starts after ply `R_open` (see the skip filter,
  §3.5).

**Adjudication & termination** (all via python-chess rules):

- No legal moves: checkmate → win for the mating side; stalemate → draw.
- Threefold repetition, 50-move rule (over the **whole game**), insufficient
  material (`board.is_insufficient_material()`, a superset of doc 11 §3.6's
  `is_material_draw`) → draw.
- **Win adjudication:** `|score| ≥ 2500 cp` for 4 consecutive plies from the
  winning side → win (matches doc 11 §3.4.7).
- **Draw adjudication:** `|score| ≤ 8 cp` for 10 consecutive plies after
  ply 80 → draw.
- **Hard cap:** game length ≥ 400 plies → draw.

All adjudication thresholds are CLI knobs so the human review can retune
them before the bulk run.

### 3.5 Curation / filtering spec

A position is **written** only if it passes every filter below. The filters
target one goal: the net learns a *static* evaluation, so the corpus must be
made of quiet, in-distribution, non-redundant positions with trustworthy
labels. These same filters bind the clean generator (doc 11 §3.4.7) and are
shared with texel tuning (doc 11 §3.3.3).

1. **Skip book/opening plies.** Do not record the first `R_open` plies of a
   game (off-teacher, near-book, low information).
2. **Not in check.** Exclude any position where the side to move is in
   check. In-check evals are unreliable and our `evaluate` is never even
   called in check (doc 11 §3.1) — such positions are off-distribution for
   the net.
3. **Quiet / non-noisy.** Exclude a position if SF's `bestmove` for it is a
   **capture, en-passant, or promotion**. The value of such a position
   hinges on an immediate tactic the *static* net cannot represent, so its
   label is noise. (This is the "quiet-labeled" filter, in the spirit of
   Texel's method; it is the single highest-leverage quality filter.)
4. **Score cap.** Exclude `|score| ≥ 2500 cp` and exclude **mate scores**.
   Near-decisive positions carry little eval-regression signal, saturate the
   WDL target, and sit outside the band our `evaluate` can even output (it
   clamps below the TB band, doc 11 §3.6).
5. **Deduplicate.** Drop exact repeats within a run, keyed by a 64-bit hash
   of (piece placement + side-to-move + castling + ep) — i.e. position
   identity, not full FEN clocks. Shuffles, repetitions, and transpositions
   would otherwise over-weight a handful of positions. A per-run `HashSet`
   is the baseline; cross-run global dedup is optional for the bulk tier.
6. **Cap positions per game.** Record at most `P_cap` positions per game
   (default **10**), sampled spread across the eligible plies, to limit
   intra-game correlation — diversity comes from *many games*, and the
   global shuffle (§3.2.2) finishes the decorrelation. (Doc 11 §3.3.3's
   "≤ 16/game" is the ceiling; 10 spreads the budget over more openings.)
7. **Phase / piece-count balance (monitor, optionally enforce).** Track a
   piece-count (2–32) and game-phase histogram across the corpus; the
   uniform-across-plies sampling of (6) plus per-game caps already gives a
   reasonable spread. The bootstrap *reports* the histogram in `meta.json`
   for human review; an optional rejection-sampling pass flattens
   over-represented buckets if the review asks for it. (Endgames are
   naturally under-sampled by self-play; this is where the enforce lever, or
   targeted endgame seeding, would go.)

### 3.6 Volume + storage

**Target counts (staged):**

| stage | positions | purpose |
|-------|-----------|---------|
| smoke | ~2 k | prove the pipeline + format end-to-end (converts, loads in bullet, python-chess round-trips) — the small validation this workflow runs before any bulk. |
| review sample | ~0.5–1 M | a short throttled run the human inspects (histograms, WDL/eval calibration curve, filter hit-rates) **before** authorising the bulk. |
| **bootstrap corpus** | **~100 M** (expandable to ~300 M) | the Tier-A training corpus for net-0. |

**Why ~100 M.** First competitive NNUE nets in the community train on ~100 M
– 1 B positions; below ~50 M a 768→2×256 net underfits and labels' noise
dominates. Doc 11 §3.4.7 already sets "≥ 100 M" as the first-net floor, and
doc 00 M9's exit criterion cites it. We anchor the bootstrap at **100 M**
(enough to prove the infra and likely to beat HCE) and pre-authorise growth
to ~300 M if net-0's SPRT vs HCE is only marginal. This is a *bootstrap*
target, not the shipped-net target (Tier B may go larger and/or self-play
"zero").

**Storage layout** (gitignored `data/nnue/`, added to `.gitignore`):

```
data/nnue/
  raw/        shard_<seed>_<worker>.txt   text shards (transient; deleted post-convert)
  bin/        shard_*.bin                 32-byte ChessBoard records
  shuffled/   shard_*.bin                 per-shard shuffled
  train.bin                               interleaved training corpus
  val.bin                                 held-out validation slice (~1%)
  meta.json                               regenerability metadata
```

- **Size.** 32 B/position ⇒ 100 M = **3.2 GB** binary; text intermediate
  ~8 GB but transient. Shard granularity ~10 M positions (~320 MB bin) for
  parallel generation, shuffle, and resumability.
- **`meta.json` (regenerability, D28's "corpus is regenerable" clause).**
  SF git rev + binary `sha256` + build ARCH; generating-engine git rev +
  its `Bench:` signature (D24); node budget; every filter and adjudication
  threshold; the CLI seed(s) and worker count; per-shard position counts;
  the piece-count / phase histogram; timestamp. Any corpus is a pure
  function of these (§3.7 determinism).

**`.gitignore`.** Add `/data/` (the corpus, gigabytes, discardable). The
SF teacher binary is already covered by the existing
`/stockfish_reference/Stockfish/` ignore. The generation *scripts*
(`tools/nnue/`) **are** tracked — they are tooling like `tools/sprt.sh`,
even though the data they emit is not.

### 3.7 Tooling & module map

The bootstrap teacher is **not** in the engine crate — it is discardable
scaffold that orchestrates an external GPLv3 binary, so it lives in
`tools/` and uses python-chess (installed) for rules/FEN/adjudication.
Python is appropriate because SF's searches dominate runtime (the harness
overhead is negligible next to 5000-node searches) and python-chess gives
correct rules for free.

```
tools/nnue/
  gen_sf_data.py     spawn local SF (UCI), self-play at fixed nodes from
                     randomized/booked openings, apply §3.5 filters,
                     emit data/nnue/raw/*.txt text shards; --seed, --nodes,
                     --workers, --games, filter/adjudication knobs.
  build_corpus.sh    convert raw→bin (bulletformat), shuffle, interleave,
                     carve val.bin, write meta.json.
  validate_sample.py sanity/format checks on a shard (§5).
```

- **Determinism / regenerability.** One SF process + one PRNG per worker,
  seeded `seed = cli_seed + worker_index · 0x9E3779B97F4A7C15`; a worker's
  output stream is a pure function of (SF binary hash, seed, worker index,
  node budget, filter + adjudication settings) — all recorded in
  `meta.json`. `--seed` is required.
- **Throttling (concurrency discipline).** The bulk run is a multi-hour
  background job that **must not starve the concurrent P9 SPRT games** on
  this 8-core box: run under `nice`/`taskset` pinned to ≤ 2–3 cores, and
  it is launched only after the curation spec is validated on the small
  sample **and human-reviewed** (D28). Nothing in this doc authorises the
  bulk run.
- **Tier B.** The clean generator is the engine's `bin/datagen.rs` (doc 11
  §3.4.7) — same format (§3.2), same filters (§3.5), no SF. It reuses the
  crate's `xorshift64star` PRNG (D15, no `rand`).

## 4. Implementation plan

Commit-sized steps; each has a done-condition. Steps 1–4 are this workflow's
"build + validate-small"; steps 5–7 are the deferred bulk + train + gate
(P13, doc 11 steps 13–14).

1. **`tools/nnue/gen_sf_data.py`** — SF UCI driver, self-play game loop,
   openings + balance gate, §3.5 filters, §3.4 adjudication, text-shard
   output, `--seed`/`--nodes`/`--workers` + knobs. *Done:* emits a
   well-formed smoke shard from a locally-built SF; every line parses.
2. **`tools/nnue/validate_sample.py`** — re-parse each emitted line with
   python-chess: FEN legal, side-to-move not in check, `bestmove` not a
   capture/promotion, `|score| < 2500`, result ∈ {0,0.5,1}; recompute
   White-relative score sign; report filter hit-rates + piece-count
   histogram. *Done:* a smoke shard passes all assertions.
3. **`tools/nnue/build_corpus.sh`** — raw→bin convert (bulletformat),
   shuffle, interleave, carve `val.bin`, write `meta.json`. *Done:* the
   smoke shard converts to 32-byte records; record count matches line
   count; `meta.json` complete.
4. **Format acceptance (small).** Load the smoke `train.bin` in bullet with
   `Chess768` + `DirectSequentialDataLoader` and run a handful of training
   steps; loss is finite and decreasing on the toy set. *Done:* bullet
   ingests our records without error — **format contract proven**. (This is
   the "validate-small" gate; no strength claim.)
5. **Bootstrap bulk run (deferred, human-reviewed).** Throttled ~100 M-
   position generation → corpus. *Done:* corpus + `meta.json`; histograms
   reviewed.
6. **Train net-0 + `netpack`** (doc 11 §3.4.7 step 3): bullet
   `768→2×256`, SCReLU, `QA=255/QB=64`, `--scale 400 --wdl 0.3`; wrap in
   CENN. *Done:* net-0 loads in the engine (doc 11 §3.4.5); symmetry green.
7. **Tier-A gate (D25):** SPRT net-0 vs Phase B HCE. *Done:* clear pass ⇒
   **NNUE infra proven** ⇒ authorise Tier B (clean self-play regeneration,
   doc 11 step 13, shipped net).

## 5. Testing and validation

- **Line format (step 2).** Every emitted line: FEN parses and is legal;
  STM not in check; `bestmove` quiet; `|score| < SCORE_CAP`; no mate score;
  result ∈ {0.0, 0.5, 1.0}. Automated over the smoke shard.
- **Perspective round-trip.** For a sample of lines, independently recompute
  `score_white` from a fresh SF search via python-chess and confirm the sign
  matches (guards §3.2.4). Confirm result is White-relative and constant
  within a game.
- **Dedup & counts.** Emitted position count == unique-key count (filter 5
  worked); `bin` record count == `raw` line count after convert.
- **Balance report.** Piece-count/phase histogram printed and stored; visual
  review that no bucket is pathologically over/under-represented.
- **Calibration curve.** On the review sample, plot binned `sigmoid(cp/400)`
  vs empirical result; near-diagonal confirms `SCALE=400` is reasonable
  (else re-fit, §9 q5).
- **bullet ingestion (step 4).** `Chess768` + `DirectSequentialDataLoader`
  reads `train.bin`; a few training steps give finite, decreasing loss.
- **Determinism.** Same seed + settings ⇒ byte-identical shard (regenerable
  from `meta.json`).
- **Strength (step 7, D25).** net-0 SPRT-beats HCE — the only claim that
  proves the *infrastructure*; no eyeballing.

## 6. Performance / throughput notes

- At 5000 nodes/move, a full self-play game is ~100 searches × 5000 nodes
  ≈ 5·10⁵ nodes; at SF's single-core Mnps this is tens of ms/game, yielding
  (after §3.5 filtering, ~10 accepted/game) on the order of **hundreds to a
  few thousand accepted positions/sec/core**. Throttled to 2–3 cores, 100 M
  positions is a **single- to low-double-digit core-hour** job — feasible as
  a background task, and the review-sample run *measures* the real rate to
  extrapolate the bulk cost before committing to it.
- Text I/O is not the bottleneck (SF search dominates); text→bin conversion
  and shuffle/interleave are one-off, I/O-bound, and run after generation.
- Disk: 3.2 GB/100 M binary; keep only `bin/` + `train.bin`/`val.bin`, delete
  `raw/` text after convert.
- The bulk run's core budget is shared with P9 SPRT — throttle is mandatory
  (§3.7), which is *why* the bulk run is gated on human review, not launched
  here.

## 7. Interactions

- **Implements** D28's curation-spec-and-pipeline clause; obeys D27 (no SF
  code/nets shipped), D15 (external tooling, no crate `rand`).
- **Feeds** doc 11 Phase C: the corpus this doc defines is what doc 11 step
  13 trains net-0 on; doc 11 owns the arch, quantization, CENN format, and
  `netpack`. The training knobs pinned here (`Chess768`, SCReLU,
  `QA/QB/SCALE`, `--wdl`) must equal doc 11 §3.4.3's constants.
- **Shares** its format (§3.2) and curation (§3.5) with the engine's clean
  self-play generator `bin/datagen.rs` (doc 11 §3.4.7) — that generator
  conforms to this doc; this doc does not re-specify its engine internals.
- **Cross-references** doc 11 §9.2 (dataset licensing), now pointed at D28 +
  this doc: NNUE-data provenance is governed here (bootstrap = discardable
  SF scaffold; shipped net = clean self-play).
- **Reuses** the strength harness (doc 14): the opening book / `gen_openings.py`
  for diversity, and SPRT (D25) for the Tier-A gate. The teacher SF is
  distinct from the packaged SF16 *opponent* under `engines/`.
- **Gated by** doc 00 M9: net-0 SPRT-beats HCE is a P13 exit criterion.

## 8. Acceptance criteria (D26)

The datagen groundwork is done when:

1. `design/15-nnue-datagen.md` specifies the exact `bulletformat::ChessBoard`
   record (§3.2.1), the text interchange line, and the White-relative
   score/result convention (§3.2.4) — reviewed against the pinned bullet /
   bulletformat version.
2. The curation spec (§3.5) enumerates every filter with rationale; the
   teacher settings (§3.3) and adjudication (§3.4) are pinned with knobs.
3. The volume plan (§3.6) states the smoke / review / bootstrap targets,
   the gitignored `data/nnue/` layout, and the `meta.json` regenerability
   set; `/data/` is in `.gitignore`.
4. The D28 progression (§3.1) is explicit: bootstrap → infra-proof gate →
   clean regeneration → shipped net trained only on our data.

The bootstrap pipeline is done (later, P13) when:

5. `tools/nnue/` generates a small validation sample that passes §5 format +
   perspective + dedup checks and is **ingested by bullet** without error
   (impl steps 1–4).
6. A ~100 M-position corpus is generated (throttled, post-review) with
   complete `meta.json`; net-0 trains and loads in the engine (CENN).
7. net-0 SPRT-beats Phase B HCE (D25) — the NNUE infrastructure is proven —
   authorising Tier B clean regeneration.
8. No SF network file or SF binary is committed or shipped; the SF-labeled
   corpus is scaffold and is deleted once Tier B lands (D27/D28).

## 9. Open questions

1. **bullet/bulletformat version pin + exact CLI.** The convert / shuffle /
   interleave invocations and the `Chess768`/`SCReLU`/`--wdl`/`--scale`
   config surface must be fixed against a specific vendored bullet +
   bulletformat commit at implementation time; their *contracts* are stable
   but flag names drift. Pin the commit in `meta.json` and the tool headers.
2. **Direct binary emission.** For the bulk tier, emitting 32-byte
   `ChessBoard` records directly (skipping the ~8 GB text intermediate)
   saves I/O — but requires replicating bulletformat's STM canonicalisation
   (§3.2.1) bit-for-bit and a round-trip equality test against the text
   path. Deferred: text is correct-by-construction and fast enough for the
   bootstrap.
3. **High-node relabel pass.** Whether to re-search accepted positions at a
   higher node budget (sharper eval targets, ~20× cost) before net-0.
   Off by default; revisit if net-0 underperforms with 5000-node labels.
4. **Binpack for Tier B.** If clean self-play corpus *storage* or
   *relabelling* dominates, adopt Viriformat/binpack (with a bullet
   `Viri`/`Sf`/`Monty` loader and in-trainer rescoring) instead of flat
   records. Not for the bootstrap.
5. **`SCALE` calibration.** `SCALE = 400` against SF's UCI-normalised cp is
   the doc 11 constant and standard; if the review-sample calibration curve
   (§5) is visibly non-diagonal, re-fit `SCALE`. One-time check when step 5
   data exists.
6. **Endgame under-sampling.** Self-play under-produces sparse endgames;
   whether to seed tablebase-ish endgame positions or enable the §3.5.7
   rejection-sampling enforce lever is a corpus-balance decision deferred to
   the review of the sample histogram.

## 10. Validation results — small sample (impl steps 1–4, 2026-07-11)

A small validation corpus was generated and verified end to end to prove the
Tier-A pipeline (§3.7, §5) before authorising the bulk run (§4 step 5). **No
bulk run was launched.** Sample: **100,071 positions** from 10,015 games,
`--seed 7`, 8 SF workers at `--nodes 5000`, random openings + ±600 cp balance
gate, all default §3.5 filters, `--p-cap 10`. Teacher: local SF
`dev-20260706-eca43a97` (sha256 `69ade9…`, ARCH `x86-64-avx512icl`). Full
`meta.json` written (`binary_status=built`).

### 10.1 Correctness — PASS

- **Format round-trip (gold standard).** `bulletformat` (v1.8.0, cloned
  read-only to scratch — GPL, never committed, D27; it has *zero* external
  deps so it builds offline) converted every text line with its own
  `ChessBoard::from_str`, and its own `DataLoader` (bullet's real read path)
  read the records back. All **99,071 train records byte-match** an
  independent re-derivation of bulletformat's side-to-move canonicalisation,
  **decode to legal boards** (python-chess `is_valid` + king/pawn-count
  checks), and bullet's own `Chess768::map_features` yields the same board with
  a consistent opponent perspective (`sq ^ 56` mirror). `build_corpus.sh` with
  that converter produced `train.bin` (99,071) + `val.bin` (1,000); its
  per-shard record==line checks passed. This meets the §4 step-4 format-contract
  gate **using bullet's actual code**, not a reimplementation.
- **Perspective / sign convention (§3.2.4).** Canonicalisation verified exact —
  for the 49.2 % black-to-move records the score is negated, the result flipped,
  and the board colour-swapped + vertically mirrored. The White-relative disk
  sign was independently re-derived by a fresh-SF resample: **100 % sign
  agreement (204/204)** for every position whose eval reaches ≥ 25 cp; the only
  disagreements sit in the dead-equal [0, 25 cp) band, where a cold re-search's
  own sign is noise.
- **Curation (§3.5).** 0 positions written in check; all `|score| < 2500`; no
  mate scores; result ∈ {0, 0.5, 1}; STM balance 50.8 % / 49.2 %. Dedup:
  within-worker 0.068 %, cross-worker global-key **0.017 %** (17/100 071) — the
  generator dedups per-worker (§3.7), so these are exactly the residue that the
  optional global-dedup bulk lever (§3.5.5) would remove; negligible. Eval
  **not saturated**: `|score| ≥ 2400` is only **0.151 %** of the corpus
  (score-cap rejects 0.19 %, mate rejects 1.4 % of searched plies); median
  15 cp, well spread across ±800. Distributions healthy — piece-count populated
  across 3..32, game-phase across 0..24 (natural full-board spike at 32/phase-24
  from openings, endgame tail present). Game result W/D/L ≈ 38.3 / 32.2 / 29.5 %.

### 10.2 Tooling fix (`validate_sample.py`)

The SF round-trip previously hard-failed on near-zero positions, because a cold
re-search's sign and bestmove diverge from the generator's in-game search
(different TT/history). Added a magnitude **deadband** (`--sign-deadband`,
default 100 cp) to the sign check and demoted the quiet-bestmove re-check to an
advisory rate — the authoritative (gen-time) bestmove the §3.5(3) filter used
cannot be re-derived from disk. Data-justified by the 100 %-above-25 cp
agreement above. (The dedup check is unchanged and correctly surfaces the
0.017 % cross-worker residue as expected, per-worker-dedup behaviour.)

### 10.3 Throughput and bulk extrapolation

- Measured **13.1 pos/s/core** steady-state (8 workers, full-box contention;
  104.6 pos/s aggregate interval) and **16.5 pos/s/core** uncontended (2-worker
  smoke). Whole-run aggregate incl. startup + opening gates: **86–90 pos/s** on
  8 workers (100 071 positions in 1 161 s).
- **§6's "hundreds to a few thousand accepted pos/s/core" is 1–2 orders of
  magnitude optimistic.** It assumed "tens of ms/game", but a fixed-5000-node
  game is ~125 SF searches ≈ 0.5 s, and `p_cap 10` writes only 10 of the ~89
  eligible quiet positions such a game already produces — throughput is
  **game-bound**, not filter-bound.
- **Bulk 100 M at `p_cap 10`:** ≈ 100 M ÷ 13–16 pos/s/core ≈ **1,700–2,100
  core-hours**. Throttled to the mandated 2–3 cores that is **~24–36 days wall**
  — not the "single- to low-double-digit core-hour" §6 implies. Even at a full
  8 workers it is ~11–13 days wall.

### 10.4 The `p_cap` lever (the key go/no-go input)

The whole cost is the ~125 SF searches/game; `p_cap` only decides how many of
the ~89 eligible quiet positions/game to *keep*. At `p_cap 10` the run
**discards 88.8 % of already-labeled positions** (`rej_pcap` 792 080 vs 100 071
written). Raising `p_cap` is near-linear *free* throughput — the searches are
already paid for:

| `p_cap` | pos/game | ≈ speed-up | 100 M cost |
|---------|----------|-----------|-------------|
| 10 (current) | 10 | 1× | ~1,700–2,100 core-h |
| 30 | 30 | ~3× | ~570–700 core-h |
| ~80 (keep most eligible) | ~80 | ~8× | ~215–265 core-h |

The tradeoff is intra-game correlation (doc 11 §3.3.3 ceiling is 16/game; the
global shuffle + many games is the decorrelator). This is the single
highest-leverage knob for the bulk run and the main thing the human review
should retune.

### 10.5 Go / No-Go

- **GO on correctness.** Format, canonicalisation, feature extraction,
  perspective, and curation are all proven against bullet's real code — the bulk
  run will not waste compute on a malformed or degenerate corpus.
- **Adjust the plan before bulk.** The 100 M / `p_cap 10` plan costs
  ~1,700–2,100 core-hours (~60× §6's estimate). Recommended before authorising
  bulk: (a) raise `p_cap` to ~30–40 (≈3–4× throughput; review the correlation
  impact on net-0), and/or (b) right-size the first bootstrap to ~10–30 M
  positions (still above the community underfit floor for a 768→2×256 net) —
  either lands the run in tens-to-low-hundreds of throttled core-hours. §6 and
  §3.5(6) should be updated with the measured rate. **Bulk generation remains
  deferred pending this human review (D28); nothing here authorises it.**
