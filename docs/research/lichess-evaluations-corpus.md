# Lichess Evaluation Corpus Contract

> **Note (2026-07-28 archive sweep):** several documents and scripts cited below were
> moved to `old/`. Prefix a broken path with `old/` to find it. Nothing in `old/` is
> loaded or trusted by the current engine; see `searchopt/LEDGER.md` for what survived.

Status: acquisition complete; semantic audit and promotion experiments in
progress, begun 2026-07-15.

## Identity and purpose

- Source: `https://www.kaggle.com/datasets/lichess/chess-evaluations`
- Kaggle handle/version: `lichess/chess-evaluations/versions/14`
- Release timestamp: 2026-01-06
- License: CC0 Public Domain
- Physical release: 17 Parquet files, 40,612,747,732 bytes
- Logical release: 342,059,879 positions and 844,812,067 flattened rows

This corpus gives Archon immediate scale and broad human-analysis-board state
coverage. It does not replace the controlled Stockfish 18 oracle. Its initial
roles are value pretraining, policy/PV pretraining, hard-position mining, and
selection of positions for deeper relabeling.

## Source semantics

The original Lichess database contains a list of evaluations for each FEN and
a list of PVs for each evaluation. The Kaggle version flattens these nested
records into rows with:

```text
fen, line, depth, knodes, cp, mate
```

Consequently, rows are not independent positions. Several adjacent rows may be
alternative PVs from one MultiPV search, and one FEN may have several searches.
Lichess recommends selecting the evaluation with the greatest depth and its
first PV when only one PV is wanted. Before Archon adopts that projection, the
audit must verify that flattened row order retains evaluation and PV order and
must define deterministic tie handling for equal depths.

Searches came from various Stockfish flavours running in user browsers. Depth
and kilo-node counts vary. Engine build, network, options, threads, aspiration
history, and exact bound status are not supplied. Labels therefore retain the
provenance name `lichess_browser_stockfish`; they are never renamed SF18 gold.

FEN strings contain only piece placement, active color, castling rights, and
en-passant. Halfmove clock, fullmove number, and repetition history are absent.
No rule-50-dependent target may silently assume a zero clock.

## Promotion gates

Before the first training smoke:

1. Verify every file's expected byte size and SHA-256 digest.
2. Verify schema and Parquet readability for all shards.
3. Establish row counts, contiguous FEN grouping, and cross-shard boundary
   behavior.
4. Determine score and mate perspective with both structural checks and pinned
   Stockfish 18 spot searches.
5. Validate first PV legality and complete-line legality on a stratified sample.
6. Quantify depth, node, MultiPV, mate, phase, castling, en-passant, and score
   distributions.
7. Measure exact and near duplicate rates; group train/validation/test by a
   stable position hash before any split.
8. Compare candidate row-selection rules on a deep SF18 audit sample.
9. Export a small streaming training shard with a machine-readable provenance
   manifest and run parser/feature/label smoke tests.
10. Promote only through held-out loss plus fixed-node engine games.

## Storage contract

Original Parquet files are immutable under
`gs://archon-main-bucket/corpora/lichess-chess-evaluations/kaggle-v14/raw/`.
Their local copies were removed only after exact-size GCS reconciliation and
the canonical export completed. Derived canonical views live in sibling
versioned directories; they never overwrite source data.
`source-metadata.json` snapshots Kaggle metadata and `download-manifest.json`
records verified local size and hash. The acquisition tool is
`tools/nnue/download_lichess_evaluations.py`.

All 17 files completed on 2026-07-15. Their combined verified size is
40,612,747,732 bytes. The collection metadata audit independently counted
844,812,067 flattened rows in 844,815 row groups; every file has the expected
schema. Nine of 16 shard boundaries bisect a FEN group. The exact audit is
`collection-audit-all17-v1.json` beside the source metadata.

## First-shard fail-fast results

The first 1,000,000 flattened rows contained 172,984 contiguous FEN groups and
316,518 distinct `(depth, knodes)` evaluation blocks. Every FEN had exactly four
fields. In this sample the first evaluation block was also the unique
highest-depth block for every position; no evaluation key recurred in a later
non-contiguous block. Of MultiPV blocks containing at least two centipawn
scores, 187,376 were ordered consistently with White-relative scores (descending
for White to move and ascending for Black) and 483 were not. This strongly
identifies White as the stored score perspective, but the exceptional blocks
still need classification. A 10,000-line legality sample passed 10,000/10,000
after trying both standard and UCI_Chess960 castling encoding.

A fast collection audit first found the issue on ten files; the completed
17-file audit confirmed identical contract schemas across all 844,812,067 rows.
Nine of the 16 file boundaries split one FEN group across adjacent shards. Any
collection exporter must therefore carry the terminal group into the next file;
processing each Parquet independently would silently split evaluations/PVs and
can select the wrong target. `tools/nnue/audit_lichess_collection.py` records
this boundary contract and will be rerun over all 17 verified files.

`tools/nnue/export_lichess_value_smoke.py` created a first 100,000-position
eval-only corpus with these preregistered filters: choose the first PV of the
maximum `(depth, knodes)` block; require depth at least 20 and at least 1,000
kilonodes; reject mates and `|cp| > 2000`; reject invalid positions and
identifiable Chess960 castling states. It produced 98,951 train and 1,049
hash-split validation rows. The result field is a neutral placeholder and the
trainer used WDL weight exactly zero. Bulletformat converted and decoded all
100,000 rows with zero piece, sign, result, or king-square mismatches.

On 988 held-out rows eligible for patched SF18 static evaluation, Lichess
searched scores versus SF18 static UCI-cp had Pearson correlation 0.797,
Spearman 0.792, and nonzero sign agreement 85.9%. The fitted SF18-static scale
was approximately `0.467 * lichess_cp - 17`; raw MAE was 83.9 cp. The scale
difference is expected between searched labels and static evaluator output and
must not be silently normalized away without engine testing.

A random-initialized classic 256-wide net trained for 2.29M presentations on
the 98,951 rows. On held-out Lichess targets it reached 97.0 cp MAE and Pearson
0.792, versus net-0 at 159.1 cp and 0.625. Thus the selected signal is readily
learnable. It nevertheless lost 0-40 to net-0 at 5,000 nodes, confirming that
100k positions from random initialization is only a parser/learnability smoke,
not a strength candidate. The next causal gate is a 1M projection tested both
from scratch and as a low-learning-rate net-0 fine-tune.

## One-million-position causal result

The deterministic projection was expanded to 1,000,000 selected positions
(989,869 train and 10,131 stable-hash validation). Bulletformat conversion and
decoding inspected all one million records successfully. A cold 30-pass net
improved held-out Lichess fit to 106 cp MAE / 0.743 Pearson, yet lost roughly
759 Elo to net-0 over 80 games at 5,000 nodes. A net-0-initialized fine-tune at
learning rate `5e-5`, WDL weight zero, improved held-out fit to 114 cp MAE /
0.677 Pearson versus net-0 at 171 cp / 0.546, yet lost roughly 407 Elo. Early
fine-tune checkpoints were also clearly negative.

A zero-learning-rate load/train/pack control produced a CENN byte-identical to
net-0 and exactly matched all 10,000 runtime probe outputs. The collapse is
therefore not an initialization or packing defect. Better imitation of these
centipawn labels is anti-correlated with Archon playing strength under this
recipe. The main missing supervision is game outcome/WDL, compounded by mixed
browser-engine provenance, absent rule-50/repetition state, correlated
analysis-board sampling, and a target scale unlike the deployed evaluator.
Consequently, this corpus is **not** promoted as a stand-alone value target.
Use it for state coverage, hard-position mining, PV/policy provenance, and
future mixed or SF18-relabeled experiments. Any value use must preserve an
outcome-bearing stream and pass early engine games.

## Complete canonical value projection

`tools/nnue/export_lichess_bulk.py` processed the 17 files as one logical
collection, including carry-over for all cross-file FEN groups. It scanned
844,812,067 flattened rows, formed 342,059,879 global FEN groups, and emitted
265,260,945 positions after the depth>=20, knodes>=1000, non-mate,
`|cp|<=2000`, and structural packing gates. The resulting 32-byte Bullet stream
is 8,488,350,240 bytes with independently verified SHA-256
`b8939f7db39cc95186ea4bae1e779741887ed6c4d96149f835baa7a60541dc21`.
It stores a neutral result byte only for format compatibility; every Lichess
value run must set WDL weight to exactly zero.
