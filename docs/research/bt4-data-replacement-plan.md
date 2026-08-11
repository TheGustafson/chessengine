# Replacing the inherited corpora with Stockfish's BT4-relabeled data (2026-08-11)

Owner direction: **replace all our old data with this; keep our SF-dev
relabeled data.**

Source: `github.com/vondele/nettest/threats.yaml` (Stockfish's live net-training
config) → the `vondele/*` and `xushawn/*` HuggingFace dataset repos. All
verified reachable, ungated, public (HTTP 200, `gated: false`).

---

## 0. The finding that drives this

**We already hold Stockfish's exact source corpora.** Our ledger records every
one of them as coming from `official-stockfish/master-binpacks`. What changed
upstream is not the positions — it is the labels. Stockfish's current config
trains on those same files carrying a `.relabel-BT4-tf13tune` suffix.

**BT4 is the Lc0 architecture** (15×1024 — the same family generating test91).
So Stockfish relabels its training data with a *Leela net evaluation*, not with
Stockfish search. That is a different labeling philosophy from the SF-dev
fixed-node search relabeling we have been doing, and it is what the strongest
engine in the world currently trains on.

**A second, unadvertised prize:** `vondele/rescored` holds ONE corpus
(`test80-2023-06-jun`) in **eight** label versions — `orig_Leela` plus
SF rescoring at n3000, n5000, n10000, n20000, n40000, n60000, n80000. That is
a published, full-scale label-depth ablation across a 27× node range. It is the
study this project ran at 250k positions locally this morning
(`label-depth-calibration.md`), done properly by the people who then chose BT4
for production.

---

## 1. Coverage map

| our corpus | mix wt | BT4 replacement | size |
|---|---|---|---|
| `uho-band75-100` | **0.1909** | `nodes5000pv2_UHO.relabel-BT4-tf13tune` | 44.2 GB |
| `farseerT74-band75-100` | 0.0920 | inside `T60T70wIsRightFarseerT60T74T75T76.split_[0-4]` | 110.5 GB (all 5) |
| `farseerT75-band75-83` | 0.0688 | ″ (same splits) | ″ |
| `multinet-band75-88` | 0.0681 | `multinet_pv-2_diff-100_nodes-5000.relabel-BT4-tf13tune` | 30.5 GB |
| `t60t70-band75-85` | 0.0625 | ″ (same splits) | ″ |
| `wisr-band75-95` | 0.0265 | `wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune` | 7.8 GB |
| **replaceable** | **0.5088** | | **193 GB** |
| `datapv2-band75-88` | 0.0738 | **none found** (`data_pv-2_diff-100_nodes-5000`) | — |
| `pylon-band75-88` | 0.0354 | **none found** (`training_data_pylon`) | — |
| `wrongnnue-band75-95` | 0.0196 | **none found** (`wrongNNUE_02_d9`) | — |
| **orphaned** | **0.1288** | | |
| `oracle-uho1` | 0.1900 | KEEP — our SF-dev in-game labels | |
| `fishpack32-sf5k` | 0.1203 | KEEP — our SF-dev 5k relabel | |
| `sfdev5k-a2` | 0.0500 | KEEP — our mining wave | |
| `canon-tight-frame`, `hardneg-frame` | 0.0022 | KEEP — ours | |
| **kept** | **0.3625** | | |

**51% of the mix is directly replaceable, 36% we keep by direction, 13% is
orphaned.**

### The orphaned 13% — three options

1. **Drop and redistribute** (simplest; loses 1.7 B positions of diversity).
2. **Relabel them ourselves with SF-dev @5k** — 1,727 M positions × $0.0261/M =
   **$45**. Cheap, and it leaves nothing inherited anywhere in the mix.
3. Keep their inherited labels (contradicts the direction).

Recommend **(2)**: $45 completes the sweep, and the per-corpus `(a,b)` /
`(k,p)` machinery already handles a mix of labelers.

---

## 2. Why this is worth doing beyond "SF does it"

Our own ledger exposes the concrete defect this fixes. Every inherited corpus
needed a **hand-audited per-corpus score divisor**:

| corpus | `SFBINPACK_SCORE_DIV` |
|---|---|
| `wrongnnue` | 1.49 |
| `pylon` | 1.95 |
| `t60t70` | 1.97 |
| `datapv2` | 2.25 |

Four different divisors, each separately audited, because each binpack's labels
came from a different engine/era/scale. That is also visible downstream in the
fitted transforms: the inherited corpora sit at `a = 142–247` while everything
we relabel ourselves lands at `a = 97–109`. **A uniformly relabeled set has one
labeler and one frame.** That is a real reduction in moving parts, independent
of whether BT4 labels are "better" than SF-search labels.

---

## 3. Open questions to resolve BEFORE bulk download

1. **What is `.q.binpack`?** Every corpus ships in two variants and the `.q` one
   is consistently **larger** (UHO 47.9 vs 44.2 GB), which rules out the obvious
   "quiet-filtered" reading. Resolve by fetching the first few MB of one file
   and inspecting the header — minutes, and it decides which 193 GB we pull.
2. **Is BT4 actually a better label than SF-dev search?** We can answer this
   *directly and cheaply*: `fishpack32.relabel-BT4-tf13tune` is only **5.9 GB**
   and we already hold the same positions in two other label versions
   (original, and our own `fishpack32-sf5k`). A three-way join on position bytes
   gives a decisive read for the price of a small download. **Do this first.**
3. **What did the rescored ablation conclude?** `vondele/rescored` at 15.8 GB
   per version. `orig_Leela` + `n5000` + `n80000` (49 GB) settles "does SF-search
   depth matter" at full scale, and cross-checks this morning's 250k-position
   local result (median 15 cp between 5k and 20k).
4. **Band filter reproduction.** Our corpora are `D40_CONVERT_BAND=75:88`-style
   filtered subsets. The BT4 binpacks are full corpora, so conversion must
   re-apply the band — and with a *uniform* labeler the per-corpus
   `SFBINPACK_SCORE_DIV` should collapse to one value, which must be re-derived,
   not inherited.

---

## 4. Disk and bandwidth

Free today: **246 GB**. Converted at our 32 B/record.

Old corpora being replaced total **218 GB** on disk, so the steady state is
roughly net-zero — but the transition needs headroom. Therefore:

**One corpus at a time, largest first, and only delete the old copy after the
new one passes its checks.** Peak extra disk = one corpus, not the set.

```
for each target binpack:
    download  ->  bf_convert (band filter, new uniform SCORE_DIV)
              ->  shuffle -> fit (a,b) -> sf_teacher_calibration
              ->  ledger add-data
              ->  verify counts + frame, THEN delete the old corpus + the binpack
```

---

## 5. Sequencing

| phase | what | cost | decides |
|---|---|---|---|
| **0** | fishpack32 BT4 (5.9 GB) → three-way label join vs original and our SF-dev 5k | ~free | **is BT4 better than our own relabeling** |
| **0b** | `.q` header inspection | minutes | which variant to pull |
| **1** | rescored ablation slice (49 GB) | free | depth question, at full scale |
| **2** | UHO (44.2 GB) — 0.191 weight, biggest single win | free | first real replacement |
| **3** | T60T70…T74T75T76 splits (110.5 GB) — covers 3 corpora at 0.223 combined | free | |
| **4** | multinet (30.5) + wisr (7.8) | free | |
| **5** | orphans: relabel datapv2/pylon/wrongnnue with SF-dev @5k | **$45** | completes the sweep |
| **6** | refit mix weights, new run spec, gate vs canonical | — | |

Phases 2–5 only proceed if phase 0 says BT4 labels are at least as good as ours.
**Phase 0 costs almost nothing and could invalidate the whole plan — so it goes
first**, and nothing bulk downloads before it reports.

---

## 6. Risks

- **Not a volume play.** We are at ~1.8 epochs over 10.26 B records. This is a
  label-quality and frame-uniformity change, not more data.
- **d63b has not yet shown that better labels move strength.** Its ladder is
  flat vs canonical through sb450. The anneal (last ~100sb) is where that gets
  answered, and the answer should inform how aggressively we commit here.
- **Licensing.** These are Lc0-derived corpora → ODbL 1.0 + DBCL 1.0, same
  obligation the audit's round 2 recorded: nets are "produced works"
  (attribution, no share-alike); **do not redistribute the corpora**.
- **DFRC** (`dfrc_n5000`, 41 GB) is available but Chess960 is deferred by D10 —
  skip it.
