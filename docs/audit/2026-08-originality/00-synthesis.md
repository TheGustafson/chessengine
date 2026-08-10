# ARCHON originality & license audit — synthesis (2026-08-09)

Three parallel domain audits (search; eval/NNUE + trainer; foundation + infra),
each with file:line receipts on both sides, against `stockfish_reference/`
at **eca43a97** (GPLv3), `reckless_reference/Reckless` (AGPLv3), and Integral
(no license; clone absent, influence audited via git history). Domain reports:
`search.md`, `eval-nnue.md`, `foundation-infra.md` in this directory.

**Verification status:** five load-bearing claims independently re-derived by
the supervising session before this synthesis was written, all confirmed:
`OUTPUT_BUCKETS_LAYOUT` verbatim vs Reckless, `partial_insertion_sort`
side-by-side vs SF, `MAGIC_SEEDS` row vs SF `attacks.cpp`, bench FEN identity
vs SF `benchmark.cpp`, and the sota_head "matched to Reckless's scalar
reference" self-documentation.

## 1. License verdict

**No obligation has triggered.** ARCHON has no LICENSE file (all-rights-
reserved by default) and has never been distributed. GPL/AGPL bind on
conveyance; rating lists and competitions count as conveyance. The decision
point is first external distribution.

Findings, most severe first:

| # | source | finding | level | receipt |
|---|--------|---------|-------|---------|
| 1 | Reckless (AGPL) | `OUTPUT_BUCKETS_LAYOUT` verbatim: identifier, all 33 values, line-grouping | **L4** | `src/eval/nnue/sota_head.rs:226` = `Reckless/src/nnue.rs:83` (VERIFIED) |
| 2 | Reckless (AGPL) | SOTA head bit-exact-matched to Reckless scalar reference; `activate`/`find_nnz` scalar kernels keep Reckless's skeleton | L2 w/ L3 elements | `sota_head.rs:2-12` self-documentation (VERIFIED) |
| 3 | SF+Reckless | `old/tools/nnue/patches/*-evalraw-v2.patch` tracked diffs carry verbatim GPL/AGPL context lines | L4 (in-repo text) | eval-nnue.md §2 |
| 4 | SF (GPL) | Architectural parity by design: SF node structure step-for-step, 259 tunables transcribed (D20), KING_BUCKETS/MAP/outer-eval constants — each element an unprotectable fact; the *wholesale curated selection-and-arrangement* is the arguable derivative-work question | L2 (judgment) | search.md §2, eval-nnue.md §2 |
| 5 | SF (GPL) | `MAGIC_SEEDS` 8 values verbatim (multipliers equal SF's by construction); bench suite 49/49 FENs byte-identical | data adoption, disclosed | `attacks.rs:639`, `bench.rs:389` (VERIFIED) |
| 6 | SF | Two functions at the L2/L3 boundary: `partial_insertion_sort` (same algorithm + identifier names, diverged expression — VERIFIED), `TtEntry::save` (rule-for-rule mirror) | L2/L3 | search.md §2 |
| 7 | Reckless | Distilled notes: five ≤10-line quoted AGPL fragments + one 5-line near-verbatim TT-cutoff quote | quotation scale | foundation-infra.md §2b |
| 8 | Reckless | 16 MB Reckless-derived PGO `.profraw`/`.profdata` git-tracked | hygiene | `reckless_reference/pgo-prof-28/` |

**Nulls that matter:** zero verbatim comment lines across 724 files (exhaustive
shingle sweep); SF notes clean at all three revisions; Integral influence nil
(13 deleted notes were pure pseudocode); Reckless expression convergence in
search and foundation: checked structure-by-structure, absent; no foreign
weights tracked; zobrist keys ≠ SF's (own draw order from SF's seed).

## 2. Recommended posture

**Remediate the Reckless items, then GPLv3 at first distribution.**

Pre-distribution checklist (≈ an afternoon + one retune). **Status column
added 2026-08-10** (remediation pass; details + post-training validation
steps in `REMEDIATION-RUNBOOK.md` in this directory):

1. Retune `OUTPUT_BUCKETS_LAYOUT` boundaries on our own data (ships naturally
   with the next net generation; head shape unchanged, only the piece-count→
   bucket map). Kills finding 1.
   **DONE-in-worktree (`e3a8b78f`)** — equal-mass layout derived from 100M
   own-corpus records (`tools/nnue/derive_output_buckets.py`, receipt
   `output-buckets-derivation.txt`); carried per-net via the new OBL1
   container section with a legacy default so every existing net is
   bit-identical (the eval-hash pin is the acceptance test, runbook step 2);
   trainer mirrored (spec/data/features/checkpoint/netpack + run key
   `arch_bucket_layout`). Compile/test deferred to the runbook (no-cargo box).
2. Restate the two ~15-line scalar kernels (`activate`, `find_nnz`) from the
   math, not the reference. Kills finding 2's L3 elements.
   **DONE-in-worktree (`89c3dcf9`)** — restated with a different
   decomposition, identical arithmetic; proven by a new randomized property
   test against the byte-untouched SIMD backends (runbook step 1 decides it).
3. Delete `old/tools/nnue/patches/` from the tree (git history is not
   distribution of the working tree; optionally scrub). Kills finding 3.
   **DONE (`1424f8c9`)** — no live references (checked incl. ledger recipes);
   no history rewrite.
4. Convert the five quoted note fragments to pseudocode. Kills finding 7.
   **DONE (`4540388c`)** — all 8 quoted/near-quoted blocks across notes
   05/07/13 converted (a superset of the five named); constants kept as facts.
5. Delete the tracked profraw. Kills finding 8.
   **DONE (`1424f8c9`)** — plus a tree-wide `*.profraw`/`*.profdata`
   gitignore so the class cannot recur.
6. Optional independence polish: pick our own magic seeds (multipliers are
   re-derived at launch; any valid seeds work, zero functional cost) and our
   own bench suite.
   **Seeds: RUNBOOK (`823c1287`)** — `find_own_magic_seeds` harness landed
   (#[ignore]d sweep + oracle validation); execution needs cargo, runbook
   step 4; shipped row untouched under a REMEDIATION-PENDING notice.
   **Bench suite: STAGED (`34af97b7`)** — 49 ARCHON-original positions from
   our own games (`src/bench_positions_archon.rs`, tool + receipt committed),
   compiled+tested but not activated: activation moves the D24 `Bench:` pin,
   runbook step 5.
7. Add LICENSE (GPLv3) + Cargo.toml license field at distribution time.
   GPLv3 moots finding 4 (the selection/arrangement question) entirely — it
   is the SF-family norm. AGPLv3 is only needed if Reckless-derived
   expression is *kept*, which the checklist removes.
   **OPEN (distribution-time owner decision; unchanged).**

## 3. Originality verdict

Honest one-paragraph answer: **ARCHON's engine core is deliberately SF-parity
— that was the S0 program — with pockets of proven original work; the net is
"SF inputs + Reckless head shape + SF outer eval"; ARCHON's genuine
originality is concentrated in its research platform and a short list of
engine mechanisms no reference engine has.**

- Search decision logic: ~85% SF-standard/SF-specific (by design), ~5%
  Reckless-idea (attributed), ~10% ARCHON-original — including the domain's
  one games-proven original win (clamped quiet SEE ordering penalty,
  +11.00 ± 7.20 Elo, LOS 99.9%).
- Eval: 8 of 24 mechanisms ARCHON-original with receipts: CENN container
  family, i8-FT with per-column scales (study 32), fully-lazy threat folding
  + mirror-refresh split (SF's source concedes its fused accumulator cannot),
  ThreatRemap learned re-coding, dis/variance/WDL heads, the JAX trainer, the
  observation store, the ledger provenance DAG.
- Foundation: commodity by nature; original infra worth presenting: oracle-
  first attacks architecture (zero-unsafe proven magic tables),
  `perft_checked` per-node invariant oracle, the ledger DAG,
  `param_provenance.py`.
- Removed-but-revivable originals (basket A / tier 0): RFP threat coupling,
  equal-value dither, followPV mode, TM/uncertainty hooks.

## 4. Differentiation program (ranked)

1. **Castling + rule50 NNUE inputs** — measured 2026-07-30: our nets AND SF's
   are bit-identically blind while SF's best move changes ~39%/~37% on twin
   positions. ~5 dense inputs, zero accumulator cost, and *nobody has it* —
   an axis to go first, not catch up. (Zero-init new columns preserves
   function → warm-compatible.)
2. **Learned counterfactual LMR allocation** (design/19) — trace →
   consequence-head → `LmrIntervention` replay loop already built; no
   counterpart in any reference engine. First experiment: train on
   LmrObservation v3 rows, validate on 3 same-mode losses, shadow SPRT.
3. **DIS head as a search consumer** — scale one margin (RFP or aspiration)
   by disagreement magnitude; the one channel that touches neither eval nor
   margins directly. Hooks live.
4. **ThreatRemap as learned FT compression** (60,720 → ~12k rows) — attacks
   the measured memory wall (fold is MSHR-limited) in a way SF structurally
   cannot retrofit; POC plumbing landed.
5. **Pin/x-ray-aware ordering signals** — the signal class study 71 opened
   (SEE mis-prices pins/x-rays); plus the two legal Reckless ideas neither we
   nor SF ship (fifty-move-bucketed correction history; root re-scoring
   mid-iteration).

**Interplay with the 700sb warm-restart plan:** keep that run DATA-ONLY (the
mix is the experiment; one change at a time). The bucket-layout retune and
new inputs belong to the *next* generation or a separate gated arm; license
remediation is independent of any training run and only gates distribution.
