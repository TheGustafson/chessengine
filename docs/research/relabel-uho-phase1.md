# Phase-1 relabel: uho-band75-100 with 5k-node eca43a97 search labels

Status: DRAFT awaiting owner go. Written 2026-08-07. Governing docs:
`data-recipe-contract.md` (D46), `sf18-label-quality-plan.md` (state boundary,
audit tiers), `cloud-journal.md` (measured cloud facts).

## Why (owner's hypothesis, and its numbers)

The eca43a97-net STATIC teacher at a 30 % blend bought ~+30 Elo (D52 chain ->
D53 promotion). 5k-node SEARCH labels dominate that on three axes: search
resolves tactics the static eval misses (SF's own search moves its eval by a
mean ~180 cp at 4k nodes — measured this session); the label applies at 100 %
weight instead of 30 %; and the per-corpus `teacher_kp` frame gymnastics
disappear for relabeled corpora. uho is 24.4 % of the training diet — the
largest single lever.

## Source and the state contract

`uho-band75-100.shuf.bin`: 2.56 B records, 82 GB, 32-byte bulletformat —
**no castling/ep/rule50 state**. The label-quality plan rules stateless
positions ineligible for deep-search labels. Resolution, MEASURED:
only **3.02 %** of a 200k sample can even possess castling rights (K+R on
home squares, an upper bound), because this is a ply-75-100 band.

**Decision: FILTER, don't mislabel.** The worker drops any record with K+R on
home squares for either side (~3 %) and relabels the rest from the
reconstructed FEN (`w - - 0 1` after STM normalisation, rule50=0). Declared
deviations from the original game state: rule50 reset (mild SF eval damping
bias at 5k nodes), ep cleared (rare and undetectable). ~2.48 B positions out.

## Method

Reuse `tools/ops/cloud/` wholesale (worker/plan/submit/fetch). New worker
stage replaces game generation with:
  read shard records -> decode FEN (mirror of `features.py` decoder, the
  `searchdelta_probe.records_to_fens` code) -> `go nodes 5000` on eca43a97
  (portable x86-64-v2 build, the SIGILL lesson) -> write records back with the
  new i16 score in SF PRINTED cp (stage-3 frame, D46), RESULT BYTE PRESERVED
  (the wdl blend needs it). D46 quiet filter re-runs at merge — expect a few
  % re-verdict shrink on top of the 3 % state filter.

Shards: ~640 x 4 M positions (128 MB in / 128 MB out per shard), ~50 min per
16-vCPU Spot task — same envelope as the oracle campaign, resumable, shards
skipped on re-submit.

## Cost (from measured constants only)

| | |
|---|---|
| rate | ~3.2 vCPU-h per 1M (derived from the c2d shard math, 250-400k nps/core) |
| compute | ~8,200 vCPU-h ≈ **$98** |
| egress | 82 GB ≈ $10 |
| wall @ measured 96-vCPU cap | **~3.6 days** |

## QA + acceptance gate (pre-registered)

* **Audit tier**: 100k random positions additionally store the full final-
  iteration info line (depth/seldepth/score/bound/nodes) per the label-quality
  plan's audit spirit; a 1k local re-search must reproduce scores exactly
  (determinism check) before the campaign scales past its canary.
* **Acceptance**: D53/D59 recipe verbatim, ONE change — uho's labels. 100-sb
  fine-tune from canonical, gates at sb50/100 vs D59's control curve
  (-13.90 / -0.00). sb100 > +10 -> extend and confirm at 1500 games;
  otherwise the relabel thesis is rejected at the cheapest possible price and
  phases 2-3 (fishpack, farseer) are NOT bought.

## Deferred, deliberately

* fishpack/farseer relabel (phase 2-3, ~$210, 7.5 days) — gated on phase 1.
* Full-state relabel from `official-stockfish/master-binpacks/` (232 GiB in
  the bucket, full game state) — the contract-clean route for fishpack if
  phase 1 pays; needs a binpack decoder we have not built.
* Quota raise past the 96-vCPU Spot ceiling — one support ticket, only worth
  filing on a phase-1 pass.
