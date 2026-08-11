# Lc0 → binpack conversion plan (2026-08-11, owner-directed)

Supersedes the ingestion half of `lc0-acquisition-plan-1B.md`; the survey
(`lc0-data-survey-2026-08-01.md`) remains the evidence base and every measured
constant below comes from it unless marked otherwise.

Owner direction: **convert to binpacks, clean it, and only take data from when
the net was strong.**

---

## 1. "When the net was strong" — mostly already solved, by Lc0's own retention

This turned out to need less filtering than expected, because Lc0 pruned its own
weak eras from storage before we ever got there.

| run | archived window | net strength across that window | verdict |
|---|---|---|---|
| **test80** (Run1) | 2024-04-01 → 2025-08-20 | plateau, Elo **205–216** the whole time (hit 208 on 2024-01-28, ended at 205) | **TAKE — all of it** |
| **test91** (Run2) | 2025-12-20 → live | seeded from a trained BT4; first net −22, worst −72, current −26 | **TAKE — all of it** |
| test79 | 2022 | **measured defect**: 63.3% SF18-d13 argmax agreement vs 73–79% elsewhere; median value error 61.8 cp vs 19–25; two instruments, n≈250 and n≈400 | **NEVER** |
| test90 | — | random-weight era (−3855 … −4145) | dir is empty; nothing to avoid |
| test75 / test60 | 2021 | older architectures, 74–76% agreement | defer |

**So the strong-net filter is a run-level allowlist, not a date cutoff:**
take test91 + test80, exclude test79. Two facts make this safe rather than lazy —
test80's archived data begins *after* its Elo plateau (everything earlier was
pruned from storage by Lc0), and test91 never had a weak phase because it was
seeded rather than cold-started (contrast run3, which starts at −4024).

**Carried caveat, unchanged:** test91-vs-test80 ranking is **not closed** — the
Elo pools are per-run with reset anchors, so −26 and 205 are not comparable, and
the survey's own SF-agreement probe is a **null** for separating the good eras
(all land 73–79%). It resolves *bad* data only. The survey's hedge recommendation
stands: **split the first slice ~50/50 and let our own training arbitrate.**

---

## 2. The central design decision: Lc0 is a POSITION source, not a label source

**Convert positions and game results; throw away Lc0's value and policy; relabel
with SF-dev at 5k nodes.**

Reasons, in order of weight:

1. **Lc0's value is MCTS Q, not an SF-style eval.** The survey's open item #1
   already flagged that our per-corpus `(a,b)` and teacher `(k,p)` machinery does
   not transfer to it. Mapping Q → cp means fitting a new transform against a
   different search's objective — an entire unvalidated layer between the data
   and the trainer.
2. **We now have the number that makes this decisive.** The label-depth
   calibration (`label-depth-calibration.md`, 2026-08-11) measured inherited
   labels sitting a **median 46 cp** from SF-dev's on identical positions. Lc0's
   labels are a *more* distant animal than the inherited SF-style ones that
   measurement was taken on.
3. **Relabeling is nearly free at this scale**: $0.0261 per 1M positions at 5k
   ⇒ **$26 per 1B**. That is an order of magnitude below the value of removing a
   whole unvalidated transform.
4. **It puts Lc0 in the same frame as everything else** — fishpack-sf5k,
   oracle-uho1 and sfdev5k-a2 all fit `a ≈ 97–109`; the inherited corpora sit at
   142–247. Same frame means the mix machinery treats it like any other corpus.
5. **The converter gets much simpler.** No Q → cp transform at all: planes →
   position, game outcome → result byte, score field left for the relabeler.

Policy is discarded outright — `pol_lambda` is 0 and the D41 policy era was
ditched — which is also what collapses the storage cost (below).

---

## 3. Storage: far cheaper than the survey projected

The survey estimated 40–100 B/position for a re-encode. That assumed keeping
sparse policy and aux fields. Discarding policy and targeting our own record
lands at exactly **32 B/position** (`archon_jax.spec.RECORD_DTYPE`: occ u64 |
pcs 16 B | score i16 | result u8 | ksq | opp_ksq | extra 3 B).

| kept positions | download (raw tars) | on disk after convert | relabel @5k |
|---|---|---|---|
| 250 M | 52 GB streamed | **8 GB** | $6.5 |
| 500 M | 104 GB streamed | **16 GB** | $13 |
| 1 B | 208 GB streamed | **32 GB** | $26 |
| 3 B | 624 GB streamed | **96 GB** | $78 |

Free disk today: **251 GB**. Nothing above 3 B needs external storage.

**Stream-convert-discard is mandatory** and unchanged from the original plan: a
tar is a plain ustar stream of independent members, so `curl | reader` parses in
one forward pass with no seeking. Peak disk = output corpus + one tar in flight
(~300 MB), never the window. A failure costs one tar.

---

## 4. Subsampling — new requirement, not in the original plan

**Lc0 records every position of every self-play game.** Consecutive records link
by exactly one legal move (the survey used this as its decode acceptance test,
709/710). That is far higher correlation than anything else in our mix: our own
pipeline caps at `PERGAME=80`, and `merge_shards.py --dedupe` exists precisely
because replayed lines carry no information.

So the converter applies a **per-game cap** at conversion time, before anything
is written. This raises diversity per byte *and* cuts the relabel bill
proportionally, because we only pay to label what we keep. Game boundaries are
detectable from the one-legal-move link breaking.

Cap to be set from a measurement, not a guess: decode one tar, histogram game
lengths, and pick the cap that matches the ~80/game our own corpora use.

---

## 5. Pipeline, in order

```
run allowlist (test91, test80; test79 NEVER)
  → stream tar (curl, no disk)
    → decode V6 (8356 B/record)
      → BYTE-REVERSAL FIX on plane bits          [acceptance test below]
      → planes → occ / pcs / ksq / opp_ksq
      → game outcome → result byte
      → per-game subsample cap                   [§4]
      → drop castle-ambiguous rows (~3%)         [relabel worker's own rule]
    → append 32 B records to shard; DELETE tar
  → relabel shards with SF-dev @ 5k nodes        [$0.0261/1M]
    → D46 trainer-parity quiet filter (default-ON at the label layer)
  → merge_shards.py --dedupe
  → fit (a,b) + sf_teacher_calibration.py, like every other corpus
  → ledger add-data with the run/window manifest in the regen recipe
```

**Acceptance test for the decoder, non-negotiable:** consecutive records must
link by exactly one legal move. The survey hit the byte-reversal trap
empirically (start position decoded as `rnbkqbnr`) before finding
`ReverseBitsInBytes` in the lc0 source; this test is what catches it. Measured
709/710 on a test91 tar.

**Manifest** (`data/nnue/lc0/manifest.tsv`, from the original plan, still the
right idea): `tar_name, bytes, sha256, positions_in, positions_kept,
converted_at, status`. Without it the next slice either re-fetches or leaves a
hole.

---

## 6. Open items and risks

1. **test91 vs test80 is unresolved** — hedge 50/50, let training arbitrate.
2. **Licensing: ODbL 1.0 + DBCL 1.0.** A net trained on it is a "produced work"
   — attribution, no share-alike. The **corpus itself must not be
   redistributed** (that would trigger share-alike). Same obligation the audit's
   round 2 flagged for the existing Lc0-derived corpora.
3. **Lc0's own rescorer** (`--deblunder`, tablebase rescoring,
   `--nnue-plain-file`) may beat writing our own reader. Worth one evaluation
   before building the decoder — it would also improve the result byte. Note it
   does NOT solve the label question: we are relabeling regardless.
4. **`test80/broken/`** was created 2026-01-19 and is now empty — someone
   quarantined part of that window and deleted it. Unexplained; matters only
   when we reach the older test80 windows.
5. **A prior `tools/nnue/relabel_lc0_uci.py` exists** but operates on normalized
   position arrays lacking full chess state (no castling/ep, white-to-move
   forced) and is explicitly marked bootstrap-only. It is NOT the path here.

---

## 7. Where this sits against relabeling what we already hold

Both are the same budget line and the same infrastructure, so they compete:

| | old corpora relabel | Lc0 slice |
|---|---|---|
| cost | $224 (all 8.58 B) | $26/B + download time |
| what it buys | better labels on positions already in the mix | **new positions** at strong-net quality |
| evidence today | median 46 cp label correction, measured | position diversity, unmeasured downstream |
| risk | low — proven pipeline | medium — new decoder, unresolved run ranking |

The honest framing: we are at ~1.8 epochs over 10.26 B records, so **neither is
a volume play**. Relabeling is a quality play on known positions; Lc0 is a
diversity play on unknown ones. d63b's ladder is currently indistinguishable
from canonical (mean of four rungs −7.2 ± 8.4), which means we do **not** yet
have evidence that either lever moves strength — and that is the argument for
sizing the first Lc0 slice small (250–500 M) rather than committing to 3 B.
