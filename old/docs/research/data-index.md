# ARCHON Data Index

**Purpose:** the single catalog of every training/label corpus ARCHON owns —
what it is, how it's laid out, when + how it was labeled, its label **scale**,
where it lives locally and in GCS, and its keep/drop verdict. This is the
human-readable companion to the machine registry `ledger/data.jsonl` (D38); the
ledger is the source of truth for hashes + regen recipes, this doc is the map.

- **Updated:** 2026-07-19 (epic housekeeping pass).
- **GCS bucket:** `gs://archon-main-bucket` (project default; the only bucket).
- **Ledger:** `ledger/data.jsonl` (31 rows) — `python3 tools/ledger/ledger.py verify`
  checks hash drift + referential integrity.
- **Rule:** metadata is git-tracked; corpus/net **bytes are gitignored** and live
  on local disk + (for the big/unique ones) GCS. Nets have **no GCS mirror yet**
  (D36 pending) — they are reproducible via `ledger/nets.jsonl` recipes.

---

## 0. Scale cheat-sheet (WHY the epic unifies on one scale)

Every legacy corpus is in a *different* unit. This is the mess the SF18-canonical
relabel (epic #40, Principle 2) exists to end. Trainer target is always
`sigmoid(cp · SCORE_MULTIPLIER / 400)`; "SF18-UCI cp" = raw Stockfish-18 UCI
`score cp` (our canonical target unit).

| Corpus family        | POV    | Native unit → SF18-UCI | Pipeline divisor | Source of the factor |
|----------------------|--------|------------------------|------------------|----------------------|
| SF18 relabel (c1/fleet) | White-rel | **1.000 (already SF18-UCI)** | **none** | recompute audit, ratio 1.000 |
| nodes5000pv2_UHO     | stm    | ÷2.82                  | ÷1.9 (→SF16 frame) | audit `uho-label-semantics-v1` |
| farseerT76           | stm    | ÷2.86                  | ÷1.9             | audit `farseert76-label-semantics-v1` (Pearson 0.934) |
| farseerT74           | stm    | ÷2.489                 | ÷1.66            | audit `farseert74-label-semantics-v1` (Pearson 0.937) |
| test80-2022-08       | stm    | (see audit)            | ~÷1.9            | audit `test80-label-semantics-v1` |
| fishpack32           | stm    | binpack score          | ÷1.9 (÷1.9 heuristic) | official SF binpack |
| lichess-evals-v14    | **White** | heterogeneous browser-SF | rejected as value target | audit `lichess-evaluations-corpus.md` |
| sf-100 (SF16 teacher)| White  | SF16-display cp        | native (÷1)      | `gen_sf_data.py` |
| lc0-test91 atanh-raw | White  | `400·atanh(q)`         | native (÷1)      | `driver_atanh.py` |

**Takeaway:** binpack labels need a per-file divisor and are stm-POV; SF18 relabel
labels are already in the canonical unit and White-relative. Mixing them without
the divisor silently corrupts training — the historical net-1/2/3 failures.

---

## 1. Official Stockfish binpacks — the full `official-stockfish/master-binpacks` set

The raw fixed-node SF/Lc0 self-play binpacks that trained every winning net —
HuggingFace `official-stockfish/master-binpacks/main`, **13 files, 287 GB**. Each
file's sha256 = its HF LFS oid (verified; matches our ledger hashes). They are
re-downloadable via `tools/nnue/fetch_official_stockfish_binpack.py`, and (2026-07-19)
the **12 standard-chess files are mirrored to GCS** (~249 GB) for a durable,
HF-independent backup + expanded relabel/training pool. **`dfrc_n5000` (Chess960)
is excluded** — we play standard chess only.

**Local** = present on `data/corpora/official-stockfish-master-binpacks/main/`.
**GCS** = present under `corpora/official-stockfish/master-binpacks/main/`
(mirror job `submit_hf_binpack_backup.py`, sha256-verified, +`.json` sidecar each).

| file | size | local | GCS | role |
|------|------|-------|-----|------|
| nodes5000pv2_UHO.binpack | 40.3 GB | ✓ | mirroring | **primary stage-1** (`uho-stage1-full`); UHO openings, 5000-node SF |
| farseerT76.binpack | 6.1 GB | ✓ | mirroring | **stage-2, trains the incumbent** (`farseert76-full`); Lc0 T76 |
| farseerT74.binpack | 20.1 GB | ✓ | mirroring | stage-2 #3 (`farseert74-full`), 3× T76 |
| test80-2022-08-…v6-dd.min.binpack | 10.8 GB | ✓ | mirroring | stage-2 rich-data (`test80-2022-08-lc0-min`), Lc0 T80 |
| wrongIsRight_nodes5000pv2.binpack | 7.3 GB | ✓ | mirroring | stage-1 sibling (`wrongisright-n5000pv2-full`) |
| fishpack32.binpack | 5.58 GB | ✓ | ✓ **done** | full official mix (`fishpack32-official`), 2.56B pos; ODbL |
| farseerT75.binpack | 47.0 GB | ✗ | mirroring | not yet local; farseer family (largest) |
| dfrc_n5000.binpack | 37.5 GB | ✗ | **EXCLUDED** | **Chess960 / DFRC** — we play standard chess only; will not use (user, 2026-07-19) |
| T60T70wIsRightFarseer.binpack | 33.0 GB | ✗ | mirroring | Lc0 T60/T70 + wrongIsRight + farseer mix |
| data_pv-2_diff-100_nodes-5000.binpack | 31.6 GB | ✗ | mirroring | pv-2 diff-100 5000-node |
| multinet_pv-2_diff-100_nodes-5000.binpack | 27.6 GB | ✗ | mirroring | multinet pv-2 diff-100 |
| training_data_pylon.binpack | 14.4 GB | ✗ | mirroring | pylon training data |
| wrongNNUE_02_d9.binpack | 5.78 GB | ✗ | mirroring (smoke) | wrongNNUE depth-9 |

Local **head/derived**: `uho-stage1-head10gib` (10.7 GB UHO prefix) — **already
reclaimed** (regen by truncation of the full UHO if ever needed). Only UHO(38G) +
wrongIsRight(6.9G) + farseerT76(5.8G) are currently on local disk; the others are
GCS-only (mirroring) + re-downloadable. **Keep** the local source files through
Phase A (needed to build the FEN manifest); drop-after-canonical is an epic open decision.
Audit + promotion gates: `docs/research/official-stockfish-binpacks.md`.

---

## 2. Lichess evaluations — bulk value / coverage (rejected as value target)

| id | what | size | positions | local | GCS |
|----|------|------|-----------|-------|-----|
| `lichess-evals-v14-canonical-265m` | Kaggle lichess/chess-evaluations v14 (CC0), projected to deepest PV per FEN | 8.49 GB | 265.3M | `data/corpora/lichess-chess-evaluations/kaggle-v14/bulk-value-v1/train.bin` | ✓ `corpora/lichess-chess-evaluations/kaggle-v14/` (45.7 GiB incl. source parquet) |

bulletformat 32B, **score-only (WDL=0)**, cp is **White-POV**, 4-field FEN.
**VERDICT: rejected** as a stand-alone value target (cp imitation anti-correlated
with strength — heterogeneous browser-SF, not SF18 gold). Retained for coverage /
policy-PV / hard-position mining / **SF18 relabel input**.
Details: `docs/research/lichess-evaluations-corpus.md`.

---

## 3. SF18 relabel campaign (the epic's canonical-data engine)

The active line. Warm/cold SF18 root labels at the true SF18-UCI scale.

| id / campaign | what | positions | depth | scale | local | GCS |
|---------------|------|-----------|-------|-------|-------|-----|
| **c1** (`sf18-uho-50k-c1-20260719`) | first UHO relabel slice | ~20M | 50k **cold** | SF18-UCI (ratio 1.000) | (pull on demand) | ✓ `sf18-relabel/c1/` (20.5 GiB; inputs+shards) |
| **c2** (held) | blanket UHO relabel | 168.5M planned | — | — | — | prefix `sf18-relabel/c2/` (held; **superseded** by Phase-A warm@20k plan) |
| `net0-paired-r5k-sf18-reckless-QUAR` | FEN + SF16 + Reckless + SF18 @5k paired | 5.41M | 5k | mixed | `data/nnue/net0-paired-reckless-d445-sf18-n5k/` | — |
| `cloud-lc0-dual-labels-r5k-QUAR` | SF18/Reckless consensus on lc0-selected | 19.99M | 5k | calibrated | `data/nnue/cloud-lc0-20m/` (80/404 shards) | ✓ `corpora/lc0-selected/v1/dual-labels-r5k-sf18-5k/` (266 shards, ~66M rows) |

**★ QUARANTINE:** the entire **5k-node** SF18 label family is quarantined from
scalar value losses — the 2026-07-15 bound audit found ~43–45% of hard-5k labels
are aspiration-window **bounds** (only ~38% jointly exact). Eligible for
policy / difficulty / bound-aware losses only. The epic's relabel is at **higher
nodes (warm@20k)** specifically to escape this.

**Phase-A target (pending Fable relabel-speed study):** warm@20k SF18 relabel of
everything-except-c1 → the canonical ~250M-position dataset. Output will land under
`sf18-relabel/<campaign>/`.

---

## 4. Legacy SF-teacher / Lc0 corpora (net-0…net-6 era — mostly superseded)

Kept for **lineage/provenance** (net-0 is still an anchor), not for new training.

| id | what | positions | size | status |
|----|------|-----------|------|--------|
| `sf-100` / `sf-100-trainbin` | SF16-teacher bootstrap (5000 nodes) — **net-0 lineage anchor** | 24.95M | 790 MB (`data/nnue/train.bin`) | **KEEP** (net-0 provenance); raw text shards already deleted, train.bin is the byte anchor |
| `sf-100-head4m` | sf-100 prefix, E1 size control | 4.06M | 130 MB | droppable (regen: head of train.bin) |
| `fishpack-sf16-relabel-4m` | fishpack POSITIONS × SF16 5000-node LABELS (E1) | 4.06M | 129 MB | keep (E1 positions-vs-labels evidence) |
| `lc0-test91` (+ `-g065`,`-atanh400`,`-atanhraw`) | Lc0 T91 A/B corpora, various Q→cp maps | ~25M each | ~1.7 GB each | **superseded** (net-1..4 era); regen from `corpora/lc0-test91/raw-tars/` (65 GiB on GCS) |
| `sf100+lc0atanhraw-50m-mix` / `-9pct-27m` | SF+Lc0 mixes (net-5/6) | 49.5M / 27.2M | 3.3 / 1.8 GB | **superseded**; symlink dirs over sources |
| `lc0-dual-100m-invalid-DELETED` | TOMBSTONE (bad move-map) | — | deleted | metadata-only |

Local roots: `data/lc0_nnue/`, `data/nnue/{mix_raw,mix90_raw,lc0-t,lc0-atanh}/`.
GCS source for regen: `corpora/lc0-test91/raw-tars/`, `corpora/lc0-selected/v1/source-arrays/`.

---

## 5. Observation-factory corpora (D41 helm; FROZEN, local-only)

RootMoveAnalysis / capture-yield shards for the learned-search (helm) line.
**FROZEN** per the path-to-1 reframe (helm is play-negative + ordering-capped).
**Local-only — NOT on GCS.** If reclaiming disk, back up to
`gs://archon-main-bucket/corpora/obs/` first.

| id | teacher | positions | size | glob |
|----|---------|-----------|------|------|
| `obs-sample-v1-{reckless,sf18}` | Reckless / SF18 | 1000 ea | ~7.8 MB ea | `data/obs/sample-v1-*.ceobs` |
| `obs-big-v1-{sf18,reckless}` | SF18 / Reckless | 10000 ea | ~77 MB ea | `data/obs/big-v1-*.ceobs` |
| `obs-gold-v1-{reckless,sf18}` | Reckless / SF18 | 20000 ea | ~57 MB ea | `data/obs/gold-v1/*.ceobs` |
| `gold-v2-teachers` | Reckless + SF18 | 100k FENs | 470 MB | `data/obs/gold-v2/*w[0-9].ceobs` |
| `obs-cap-yield-pilot` | s3-farseer self-obs | 644k | 376 MB | `data/obs/harvest/yield-probe/*.ceobs` |
| `capture-scale-v1-train` / `capture-scale-batches-v1` | s3-farseer self-obs | 41.5k / — | 76 MB / 829 MB | `data/obs/train/capture-*` |

Total `data/obs/` ≈ 2.3 GB.

---

## 6. GCS bucket layout (`gs://archon-main-bucket`) — post-2026-07-19-cleanup

```
corpora/
  official-stockfish/master-binpacks/main/   ~249 GiB  12 standard binpacks (mirroring; ex-dfrc-960) +.json sidecars
  lc0-test91/raw-tars/                        65 GiB    raw Lc0 T91 tarballs — DEAD line, pending delete decision
  lc0-selected/v1/                            23 GiB    source-arrays + position-shards (dual-labels DELETED); dead policy line, pending delete
  lichess-chess-evaluations/kaggle-v14/       7.9 GiB   bulk-value-v1 export + manifests (raw parquet DELETED — Kaggle-recoverable)
sf18-relabel/
  c1/                                         20.5 GiB  inputs/{manifest.json,tasks/*.fens} + shards (c2 DELETED — abandoned plan)
nets/                                         2.4 GiB   shipped-net backup (incl. incumbent .cenn) — NEW 2026-07-19
benchmarks/                                   15 MiB
```

**Cleanup 2026-07-19 (~46 GiB freed):** deleted `sf18-relabel/c2` (abandoned),
`lc0-selected/…/dual-labels-r5k-sf18-5k` (quarantined 5k), `lichess/…/raw` (parquet,
re-downloadable), `smoke/`. **Pending user confirm:** the dead lc0 line
(`lc0-test91/raw-tars` + `lc0-selected/v1/{source-arrays,position-shards}` ≈ 88 GiB).

---

## 7. Local disk map + reclaim candidates (root `/dev/sdc`, 251 GB, ~94% used)

`data/` = 70 GB: `corpora/` 51 GB (§1–2 sources) · `nnue/` 17 GB · `obs/` 2.3 GB · `lc0_nnue/` 266 MB.

**Reclaim status/priority (safe = superseded/regenerable AND not incumbent-lineage):**
1. ✅ Dead-run intermediate checkpoints (`*-sbNN-{raw,quant}.bin`, non-final) — **20 files / 2.1 GB deleted 2026-07-19** (finals + backed-up `.cenn` kept).
2. ✅ `uho-stage1-head10gib` (10.7 GB) — **already reclaimed** (prior session).
3. Superseded lc0/mix text corpora (§4) — droppable, regen from GCS raw-tars (but the lc0 line itself is pending a delete decision — §6).
4. After Phase-A manifest is built: local raw binpacks (UHO/wrongIsRight/farseerT76 ≈ 51 GB) become GCS-redundant → drop for the local retrain headroom.

**Never delete:** `data/nnue/nets/*.cenn` (shipped nets), the incumbent
`stage2-v1-s3-2048-farseer-120sb` lineage, `data/nnue/train.bin` (net-0 anchor),
Phase-A source binpacks (§1) until the canonical set exists.

---

## 8. Backup & retention policy

- **Corpora (§1–4):** big/unique ones live on GCS; official binpacks are
  re-downloadable (recipe = backup). Verify a GCS copy exists *before* deleting a
  local corpus that isn't trivially re-downloadable.
- **Nets:** local-only today (D36 GCS mirror pending). The **shipped `.cenn`
  files are the product** — mirror `data/nnue/nets/*.cenn` to
  `gs://archon-main-bucket/nets/` as cheap insurance (they are ~3 MB each).
- **Obs (§5):** local-only, frozen; back up to `corpora/obs/` before reclaiming.
- **Every** new corpus/net/relabel campaign → a `ledger/data.jsonl` (or
  `nets.jsonl`) row with a regen recipe, per D38.
