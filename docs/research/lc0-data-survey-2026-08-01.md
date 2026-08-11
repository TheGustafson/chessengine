# Lc0 public training-data archive — survey and acquisition plan

**Date:** 2026-08-01/02 · **Archive:** `https://storage.lczero.org/files/training_data/`
**Status:** first-hand measured. Every volume number below was derived from the live
directory listings; every format claim was verified by decoding real records pulled
from the archive; the strength timeline comes from `training.lczero.org`'s own
network tables (primary source).

**License:** the data is **ODbL 1.0** (database) + **DBCL 1.0** (contents) —
`https://storage.lczero.org/files/training_data/LICENSE.txt` (fetched 2026-08-01,
verbatim). ODbL is share-alike *for the database*; individual contents are DBCL.
A net trained on it is a "produced work", not a derivative database — but read the
licence before publishing a corpus derived from it.

---

## 1. ACQUISITION PLAN (ranked)

Constants used throughout, both **measured**, not assumed:

| constant | value | how measured |
|---|---|---|
| bytes per position, tar level | **208 B** | 7 samples across 6 eras: 205.8 / 207.1 / 207.4 / 211.3 / 211.3 / 211.9 / 212.0 |
| positions per GB of tar | **4.81 M** | 1e9 / 208 |
| download throughput | **25.9 MB/s** (93 GB/h) | `curl -r 0-104857599` on a test91 tar, 2026-08-01 23:58 UTC, from this box |

**Bottom line: 1 B positions = 208 GB = 2.2 h. 10 B positions = 2.08 TB = 22 h.**
The archive holds **~135 B positions** in total, of which **~51.6 B** are from the
two runs I can positively certify as strong.

### Rank 1 — `test91`, ALL of it (the whole live run)

```
https://storage.lczero.org/files/training_data/test91/training-run2-test91-YYYYMMDD-HHMM.tar
```

| | |
|---|---|
| Span | 2025-12-20 → 2026-08-02 (**live**, +~4.0 GB/day) |
| Size | **1,375.7 GB** across 4,590 non-empty hourly tars |
| Positions | **6.61 B** |
| Download | ~14.8 h |
| Generating net | **15×1024 (BT4-class)**, run2 nets `910830`–`913222` |
| **Visits per recorded position** | **fixed ~800** (p1=800, p99=831) — measured |
| Policy | yes, full 1858-float MCTS visit distribution |

**Why it is strong.** Three independent reasons, all checkable:

1. It is the *only* active Lc0 run (`training.lczero.org/training_runs`, run 2,
   "Another debug run for training BT4 in RL mode", status **Active**), and it uses
   the **BT4** architecture. Lc0's own best-nets page
   (`https://lczero.org/dev/wiki/best-nets-for-lc0/`, "The networks below are our
   strongest available") heads its list with **BT4-1024x15x32h**, above
   BT3-768x15x24h and T82-768x15x24h. The training server reports test91's nets as
   **15×1024** — the BT4 shape — where test80's final nets are **15×768**, the
   BT3/T82 shape. Independently, the network blobs at
   `storage.lczero.org/files/networks/` step from a flat 170 MB (2024-01 → 2025-08,
   the whole test80 window) to 320–370 MB from 2025-09 on.
2. **It was seeded from a trained net, not from scratch** — this is the single most
   important thing to establish about a new run, and it is directly visible in the
   run's own Elo table. run2's first test91 network `910019` (2025-11-07 08:06 UTC)
   is rated **−22** against the run anchor. Compare run3/test90, which genuinely
   started from random weights the month before: its first nets are rated **−4024**
   (`training.lczero.org/networks/3?show_all=1&page=4`). A fresh run looks like
   −4000; test91 never did.
3. **Every recorded position carries a full ~800-visit search.** T80 and every
   earlier run used KLD-gain early stopping, so their recorded positions have a
   *median* of ~470 visits with a floor at 200. Per position, test91 labels are
   backed by roughly **1.7× the search** of T80's. Measured, not inferred — and it
   agrees exactly with the runs' published self-play parameters (`training_runs`
   "Train Params"): T80 = `--visits=10000` **with**
   `--minimum-kldgain-per-node=0.000050`, T91 = `--visits=800` with **no KLD gain
   term at all**. Two independent sources, same answer.

   Also worth knowing: BT4's measured advantage over the architecture family T80
   ended on is large. Lc0's own numbers
   (`https://lczero.org/blog/2024/02/transformer-progress/`): policy Elo relative to
   T78 — BT3 **+179**, **BT4 +270**. And `BT4-spsa-1740` is the net Lc0 sends to
   TCEC and CCC (`https://github.com/LeelaChessZero/lc0/wiki/Networks`).

Sub-windows if you want to stage it (contiguous, no gaps > 2 days):

| window | GB | positions | dl |
|---|---|---|---|
| 20260709–20260801 | 132.9 | 0.64 B | 1.4 h |
| 20260421–20260627 | 286.5 | 1.38 B | 3.1 h |
| 20260314–20260413 | 78.6 | 0.38 B | 0.8 h |
| 20260101–20260311 | 697.3 | 3.35 B | 7.5 h |
| 20251220–20251229 | 178.4 | 0.86 B | 1.9 h |

Note the run's Elo is **flat** across all of these (−12 to −33 over the whole
archived window, `training.lczero.org/networks/2?show_all=1&page=1..30`), so there
is no quality reason to prefer one window over another. Take them in whatever order
is operationally convenient; recency buys nothing here.

### Rank 2 — `test80`, final window `20250709`–`20250820`

```
https://storage.lczero.org/files/training_data/test80/training-run1-test80-YYYYMMDD-HHMM.tar
```

761.9 GB · **3.66 B positions** · 1,008 tars · ~8.2 h.
Generating net: run1 nets ~`825800`–`826344`, **15×768**, Elo **205–207**.

### Rank 3–6 — the rest of `test80`, newest first

| window | GB | positions | tars | dl | net Elo in window |
|---|---|---|---|---|---|
| 20250306–20250427 | 1,193.4 | 5.74 B | 1,243 | 12.8 h | 204–208 |
| 20250125–20250303 | 825.2 | 3.97 B | 870 | 8.8 h | 204–206 |
| 20241119–20250103 | 1,193.8 | 5.74 B | 1,072 | 12.8 h | 198–209 |
| 20240401–20241105 | 5,386.4 | **25.90 B** | 5,193 | 57.8 h | 203–210 |

**test80 total: 9,364.8 GB = 45.02 B positions.**

**The strongest single argument for taking test80: BT4 was trained on it.** Lc0
maintainer borg323, `https://github.com/LeelaChessZero/lc0/discussions/1981`:
*"BT3 and BT4 were trained using data generated by our training pipeline for T80."*
BT4 is the net Lc0 sends to TCEC/CCC. The archived window is a **subset of the same
data that produced Lc0's strongest network** — it is not merely "plausibly good",
it is demonstrably sufficient to train the strongest thing they have.

Two structural facts about test80 that matter and are easy to get wrong:

- **T81 and T82 are phases of T80, not separate runs.** The generating architecture
  changed twice *in place*: 15×512 → 19×512 at net `808336` (2023-01-05), then
  → **768×15×24h transformer** at net `811972` (2023-05-07 ~02:17, pinned by the
  config archive: `training_configs/run1/512x19-t80.yaml-20230507-0216` last write,
  `768x15x24h-t80.yaml-20230507-0218` first write). **Both switches predate the
  archived window by ≥11 months**, so all 45 B archived positions come from the
  768×15×24h transformer generator — one architecture, no switch artifacts.
- The recommended best-net `T82-768x15x24h-swa-7464000`
  (`https://lczero.org/dev/wiki/best-nets-for-lc0/`) is a **run1/T80 artifact**,
  exported 2024-02-13 — i.e. from this run, at its peak.

**Why all of it is strong — the single cleanest fact in this document.** Lc0
pruned test80's early data. What remains in the archive begins **2024-04-01**, and
the run's own Elo table shows T80 reached its plateau on **2024-01-28** (net
`817345`, Elo 208) and stayed at **203–210** until the run ended on **2025-08-20**
(final net `826344`, Elo 205, 15×768 — `training.lczero.org/networks/1?show_all=1`,
row 1, timestamp `2025-08-20 13:11:37`). The archived window sits *entirely* on the
plateau. The measured curve, from `training.lczero.org/networks/1?show_all=1&page=N`:

```
net      Elo   date         arch
800001     0   2022-04-04   15x512   <- run start ("Main run started April 4th 2022")
800345    59   2022-04-22   15x512
802345   107   2022-07-01   15x512
804345   124   2022-09-02   15x512
806345   131   2022-11-08   15x512
808345   127   2023-01-05   19x512
810345   149   2023-03-19   19x512
812345   158   2023-05-17   15x768
814345   179   2023-07-19   15x768
816345   198   2023-11-21   15x768
817345   208   2024-01-28   15x768   <- plateau reached
818345   210   2024-03-30   15x768   ]
820345   209   2024-07-05   15x768   ]  ARCHIVED DATA WINDOW
822345   203   2024-11-22   15x768   ]  (2024-04-01 .. 2025-08-20)
824345   208   2025-04-03   15x768   ]
826344   205   2025-08-20   15x768   ]  <- run ends
```

Within the archived window the spread is 198–210, i.e. **within the Elo table's own
resolution**. There is no "early weak period" left in test80 to avoid; the pruning
already removed it. Rank the windows by recency only because it costs nothing to.

**One honest qualification.** T80's *peak* self-play rating is **216 at net `817760`,
2024-02-23** — about five weeks *before* the archived data begins. So the archived
window is not "the peak"; it is the eighteen-month **post-peak plateau** (198–211).
The run stopped improving in early 2024 and generated for another 18 months at a
flat level. For a data-acquisition decision this is fine — flat means every window
is equivalent — but do not repeat "T80's best data" as if the peak were inside the
window. It is 5 weeks outside it, and the data from the peak is deleted.

### Rank 7 — `test60` (2019-07-26 → 2022-01-08)

5,724.4 GB · **27.52 B positions**. Format V6 from ~2021 on; **the 2020 tars are V5
(8308-byte records)** — verified: `training-run1-test60-20200601-0017.tar` and
`training-run3-test71-20200601-0022.tar` both decode at stride 8308, while
`training-run1-test60-20211201-0017.tar` is V6/8356. Generating nets are the T60
line, ending at 384×30. Take only if you have exhausted ranks 1–6 and want raw
diversity; the nets are two architecture generations behind.

### Rank 8 — `test75` (2021-03-14 → 2021-11-28)

6,129.8 GB · **29.47 B positions**. V6. Same caveat as test60, plus it scored
slightly worse than test80/test91 on my value probe (median |lc0 − SF| 29.4 cp vs
19–25 cp). Low confidence in that difference — see §5.

### DO NOT TAKE

| path | why |
|---|---|
| `test79/` (2,668 GB, 12.83 B pos) | **Measurably the worst era I sampled.** SF18 d13 agreement 63.3% vs 73–79% everywhere else; median value error **61.8 cp** vs 19–25 cp; mean cp-loss of the played move 13.8 vs 2.1–9.1. Two independent instruments, n=254 and n=251, both flag it. It is a 4-month 2022 run that was replaced. |
| `test78/`, `test90/` | **Zero bytes.** Every tar is a 10,240-byte empty archive (4,086 and 557 of them respectively). test90 is run3, "Debug run for training BT4 in RL mode", **Inactive**, and its Elo curve starts at **−4884** (random weights, 2025-09-30) and only reached −262 before abandonment on 2025-11-06. Even if data appeared, it would be from-scratch garbage. |
| `training_data/run3/` (2019 tars) | V4 format (8292-byte records), 2019 nets. |
| `training_data/training-run{1,2,3}--*.tar` (top level) | Mostly 10,240-byte empty archives; the few real ones are one-off snapshots. |
| `test71/`, `test71_5/` | 237 GB total, 2020, superseded. |
| `training_pgns/` | **Empty directory** (verified 2026-08-01). |

### Recommended concrete order

```
1.  test91 entire          1,376 GB    6.61 B pos   ~15 h   <- start here; >1B in 2.2h
2.  test80 20250709-0820     762 GB    3.66 B pos    ~8 h
3.  test80 20250306-0427   1,193 GB    5.74 B pos   ~13 h   [cumulative 16.0 B]
4.  test80 20250125-0303     825 GB    3.97 B pos    ~9 h
5.  test80 20241119-0103   1,194 GB    5.74 B pos   ~13 h   [cumulative 25.7 B]
6.  test80 20240401-1105   5,386 GB   25.90 B pos   ~58 h   [cumulative 51.6 B]
```

**Disk discipline (this box has ~126 GB free and a trainer running).** Never
materialise a window. Stream one tar at a time:

```bash
url=.../training-run2-test91-20260801-2317.tar
curl -sL "$url" | our_reader --stdin --out shard.bin   # ~180 MB in flight, nothing on disk
```

A tar is a plain ustar stream of independent gzip members — it parses in one
forward pass with no seeking, so `curl | reader` works and peak disk is whatever
your output format costs. At 208 B/position raw, a compact re-encode (see §3.7)
brings 1 B positions from 208 GB to ~40–100 GB.

---

## 2. STRENGTH TIMELINE

### The archive's own structure (measured 2026-08-01/02)

`https://storage.lczero.org/files/training_data/` — every subdirectory, complete
(I probed `test70/74/77/81/82/83/84/85/86/87/88/89/92` — all **404**; the listing
is exhaustive):

| dir | GB | positions | non-empty span | run | status |
|---|---|---|---|---|---|
| `test60` | 5,724.4 | 27.52 B | 2019-07-26 → 2022-01-08 | run1 | ended |
| `test71` | 237.2 | 1.14 B | 2020-04-10 → 2020-07-04 | run3 | ended |
| `test75` | 6,129.8 | 29.47 B | 2021-03-14 → 2021-11-28 | run3 | ended |
| `test78` | **0.0** | 0 | — (4,086 empty tars 2025-03→2025-08) | run3 | no data |
| `test79` | 2,668.1 | 12.83 B | 2022-02-25 → 2022-05-06 | run2 | ended |
| `test80` | 9,364.8 | 45.02 B | 2024-04-01 → 2025-09-24 | run1 | **ended 2025-08-20** |
| `test90` | **0.0** | 0 | — (557 empty tars 2026-07-09→2026-08-02) | run3 | abandoned |
| `test91` | 1,375.7 | 6.61 B | 2025-12-20 → **2026-08-02** | run2 | **LIVE** |
| `test40` | 1,387.8 | 6.67 B | 2019-01-08 → 2019-07-26 | — | ended (2019 nets) |
| `test30` | 1,046.0 | 5.03 B | 2018-09-09 → 2019-02-02 | — | ended (2018 nets) |
| `test71_5` | 125.0 | 0.60 B | 2020-07-05 → 2020-08-30 | run3 | ended |
| `run3` | 83.5 | 0.40 B | 2019-06-18 → 2019-08-27 | run3 | ended, **V4 format** |
| **total** | **28,142** | **135.3 B** | | | |

(`test30`, `test40`, `run3`, `test71_5` are 2018–2020 eras, two to four architecture
generations behind, and are not considered further.)

The three "runs" are hosting slots, not eras: run1 hosted test60 then test80; run2
hosted test79 then test91; run3 hosted test71 → test75 → test78 → test90.

### The run register (primary source)

`https://training.lczero.org/training_runs`, fetched 2026-08-01 — verbatim:

| Run ID | Name | Description | Status | Best Network |
|---|---|---|---|---|
| 1 | Run1 (test80) | "Main run started April 4th 2022 (Partially pre-trained)" | **Inactive** | 826344 |
| 2 | Run2 (test91) | "Another debug run for training BT4 in RL mode" | **Active** | 913222 |
| 3 | Run3 (test90) | "Debug run for training BT4 in RL mode" | **Inactive** | 900434 |

Confidence **HIGH** — this is Lc0's own training server.

### T80 — strong, plateaued, and the weak part is already deleted

Evidence table in §1 Rank 3–6. Sources:
`https://training.lczero.org/networks/1?show_all=1&page=N` for N ∈ {1,5,…,264},
and the Elo series embedded in `https://training.lczero.org/training_run/1?full_elo=1`
(26,346 points, rating range −12 … 216).

Key dates, all from those pages:
- **2022-04-04** run starts, "partially pre-trained", net `800001` Elo 0.
- **2023-01-05** architecture 15×512 → 19×512 (net `808345`).
- **2023-05-17** → 15×768 (net `812345`), the shape it keeps for the rest of the run.
- **2024-01-28** Elo 208 — plateau.
- **2024-04-01** archived data begins (everything earlier was pruned from storage).
- **2025-08-20 13:11 UTC** final net `826344`, Elo 205. **Run ends.**
- The `test80/` directory keeps emitting 10,240-byte empty tars from 2026-07-09
  onward; `test80/broken/` exists and is **empty**.
- Storage also holds three stray real tars dated 2025-09-10/24 (≈2 GB) after the
  run's last net.

Architecture cross-check from `https://storage.lczero.org/files/networks/`: net
blobs are a flat **170 MB** every month from 2024-01 through 2025-08 and then jump
to 320–370 MB in 2025-09 — exactly the run1-ends / BT4-begins boundary. Independent
of the Elo table, and it agrees.

### T91 — seeded, flat, BT4, live

`https://training.lczero.org/networks/2?show_all=1&page=N`:

```
910019   -22   2025-11-07 08:06   15x1024   <- first test91 net
910119   -65   2025-11-17 10:42   15x1024
910219   -57   2025-11-23 15:05   15x1024
910723   -53   2025-12-11 18:34   15x1024
        ---- archived training data begins 2025-12-20 ----
911223   -15   2025-12-30 06:51   15x1024
911723   -11   2026-01-30 21:01   15x1024
912223   -19   2026-03-23 15:41   15x1024
912723   -16   2026-04-10 02:47   15x1024
913123   -25   2026-07-12 18:57   15x1024
913222   -26   2026-08-01 23:43   15x1024   <- current best net
```

Read this carefully, because it cuts both ways and I am not going to sand it down:

- **Good:** the run never went through a weak phase. First net −22, worst −72, now
  −26. It was seeded from a trained BT4, which is exactly what "RL mode" implies and
  what the contrast with run3 proves. **All archived test91 data was generated by a
  net within ~50 Elo of that seed.** There is no "skip the first N weeks" advice to
  give, and in fact the first six weeks (2025-11-07 → 2025-12-19) are not in the
  archive at all.
- **Bad / honest:** the run is **not improving**. Nine months in, its best net is
  rated slightly *below* its own anchor. It is labelled a "debug run". Do not buy
  the story that test91 data is strong *because the run is progressing* — it isn't.
  Buy it because of what the net **is** (BT4, 15×1024, the architecture Lc0 sends to
  TCEC — `https://lczero.org/dev/wiki/best-nets-for-lc0/`) and because of the
  **measured 800 visits per position**.

### T90 — the counterexample that validates the method

`https://training.lczero.org/networks/3?show_all=1&page=4`: run3's nets `900035`–
`900134` (2025-10-01 → 2025-10-10) are rated **−3855 to −4145**. That is random
weights. The run climbs to −262 by 2025-11-06 and is then abandoned. This is what
"a fresh run" looks like in these tables — and neither test80's archived window nor
test91 looks anything like it. Its data directory is empty, so nothing to avoid.

### T79 — the one era with a positive quality *defect*

Not just "older", **measurably worse**, on two independent instruments (see §5 for
the null discussion):

| era | SF18 d13 argmax agreement | mean cp-loss of played move | median &#124;lc0 cp − SF cp&#124; | decisive-sign agreement |
|---|---|---|---|---|
| test91 2026-08 | 75.0% (n=232) | 8.7 | 25.3 | 96.4% |
| test91 2025-12 | 75.2% (n=226) | 5.5 | 18.7 | 98.4% |
| test80 2025-08 | 78.9% (n=266) | 3.3 | 22.0 | 97.0% |
| test80 2024-04 | 73.3% (n=258) | 3.6 | 19.1 | 93.4% |
| test60 2021-12 | 75.6% (n=242) | 2.1 | 18.6 | 90.3% |
| test75 2021-09 | 74.0% (n=273) | 8.9 | 29.4 | 92.7% |
| **test79 2022-03** | **63.3% (n=251)** | **13.8** | **61.8** | **87.6%** |

Method: decode planes → FEN; keep positions with ply ≥ 16–24 and
`played_idx == best_idx` (so the move played *is* the search's choice, removing
self-play temperature noise); recover the played move by diffing consecutive
positions; score with `engines/bin/stockfish18` at fixed depth 13, 1 thread, mate
clamped to ±1500 cp; `searchmoves` for the played move's own score. Q→cp via Lc0's
own `90·tan(1.5637541897·q)` (`src/search/classic/search.cc:322`).

### Retention

`test80`'s data starts 2024-04-01 although the run started 2022-04-04, while
`test60` (2019!), `test75` and `test79` are intact. So retention is **not** a
uniform time window — early test80 was deleted deliberately. I could not find an
announcement saying so. Whatever the reason, the effect is the one described above:
what survives is the plateau. Confidence in the *fact*: HIGH (directory listing).
Confidence in the *reason*: none, I did not establish it — see §5.3.

One more retention artifact worth knowing when scripting a fetch: the empty
10,240-byte tar is what an **idle run slot** emits, not what deletion leaves behind.
That is why `test78` (2025-03→2025-08) and `test90` (2026-07→2026-08) are entirely
empty — run3 was idle under those names — and why `test80` keeps emitting empties
after 2025-09-24. Skip anything ≤ 100 KB; do not treat it as a hole in the data.

### Data-generation settings I could measure directly

| era | visits/position p1 / p50 / p99 | mean | policy KLD | played==best | mean game plies | draws |
|---|---|---|---|---|---|---|
| test91 | 800 / 808 / 831 | **810** | 0.13 | 60.1% | 115–125 | ~56% |
| test80 | 201 / 460–473 / ~1000 | 483 | 0.16 | 64–66% | 107–109 | ~51% |
| test79 | 201 / 469 / 1096 | 506 | 0.24 | 61.8% | 110 | ~24% |
| test75 | 201 / 535 / 1241 | 568 | 0.22 | 65.5% | 99 | ~33% |
| test60 | 205 / 654 / 1653 | 706 | 0.24 | 68.2% | 113 | ~56% |

test91 is the only era with a **fixed** visit count — everything else uses KLD-gain
early stopping (visible as the 200 floor and the long right tail). Also measured:
**2–5% of games in every era start from a shuffled back rank (DFRC)** and are
recorded with `input_format = 1`, which cannot express rook files — those games'
castling rights are not reconstructable. Detect them by checking record 0's back
rank and drop them, or accept ~3% ambiguity.

`invariance_info` histogram across all eras: ~75% of positions carry **bit 5 =
"game adjudicated"**, ~15% zero, ~5% bit 3 = "best_q is for a proven best move".
So most game results are adjudication results, not played-out results.

### The runs' published self-play parameters

From `https://training.lczero.org/training_runs` ("Train Params" column). These are
the settings that produced the visit distributions measured above — and they agree:

| | **T80 (run1)** | **T90 (run3)** | **T91 (run2)** |
|---|---|---|---|
| visits/move | 10000 *cap* | **400** | **800** |
| KLD early stop | `--minimum-kldgain-per-node=0.000050` | **none** | **none** |
| smart pruning | default | `0.0` (off) | `0.0` (off) |
| cpuct / at-root | 1.20 / 2.0 | 1.20 / 1.414 | **0.65** / 2.53 |
| FPU reduction | 0.49 | 0.49 | **0.23** |
| policy softmax temp | 1.45 | 1.45 | 1.45 |
| Dirichlet noise | ε 0.1, α 0.12 | same | same |
| temperature | 0.9 → 0.60 endgame, decay 60, delay 20, cutoff 40, visit-offset −0.8 | same | same |
| resign | 2.0%, playthrough 20 | 1.0% | 1.0% |
| opening book | `run2_book.pgn.gz`, shuffled | same | same |

The T80 `10000` is a *cap*, not the realised count — KLD gain stops the search far
earlier, which is exactly the 201/470/1000 distribution I measured. **T90 and T91
dropped the KLD term entirely** in favour of a fixed low visit count: a deliberate
quality-per-position-vs-volume trade, and the reason test91 positions are uniformly
800-visit.

Note also the temperature schedule: moves before ply ~40 are sampled, not chosen
greedily. That is why `played_idx == best_idx` is only 60–68% and why any
move-quality probe must filter on it (as mine does).

### The archive is RAW self-play — the rescorer is a separate step, and we can run it

`invariance_info` bit 6 is documented as *"position marked for deletion by the
rescorer (**never set by lc0**)"* (`src/trainingdata/trainingdata_v6.h` L52), and
`orig_q/orig_d/orig_m` are annotated *"For value repair"* and are NaN when the eval
was not cached. So what is on the server is lc0's raw self-play output; Lc0's own
training pipeline rescores it before training (T80's very first config already lists
`shadow1/data-rescored/`, `-rescored2/`, `-rescored3/` in `input_train` —
`storage.lczero.org/files/training_configs/run1/512x15.yaml-20220404-0248`).

**This is an opportunity, not a caveat.** `src/trainingdata/rescorer.cc` exposes:

- `--syzygy-paths`, `--gaviotatb-paths` — **tablebase rescoring of the value target**
  (we have TB shims already),
- `--deblunder` with `--deblunder-q-blunder-threshold` / `-width` — *"use move Q
  information to infer a different Z value if the selected move appears to be a
  blunder"*, i.e. repair outcome labels corrupted by a single self-play error,
- `--policy-substitutions`, `--temperature`, `--dist_offset`, `--dtz_policy_boost` —
  policy-target reshaping,
- **`--nnue-plain-file`, `--nnue-best-score`, `--nnue-best-move`** — an export path
  that writes Stockfish-NNUE *plain* format directly.

That last one means Lc0 ships a supported route from this archive into an
NNUE-trainer input format. Worth evaluating before writing our own reader end to end.
(Not verified by me at runtime — flag names read from the source, not executed.)

### Why test91's hourly tars shrank 900 MB → 180 MB

Not a quality change — a **contributor collapse**. `https://training.lczero.org/`
currently reports *3 users in the last day* having played 162,711 games, of which
160,447 come from a single contributor. T80 in 2025 consumed ~1 M games/day; test91
now produces ~1/6 of that. Consequence for us: test91's 6.6 B positions are the
output of a very small number of machines, so hardware/timing diversity is low —
but the visit count is fixed, so label quality is not affected.

### `t82lionbroken/` and `t82wrongactbroken/`

Both hold `768x15x24h-t82-*.pb.gz` checkpoints from 17–20 April 2023 — two failed
attempts to train the T82 network (one with the Lion optimizer, one with a wrong
activation), abandoned; the third attempt succeeded and its series continues from
`768x15x24h-t82-swa-29000.pb.gz` (2023-04-20). These are **network** directories,
not training data, and they predate the archived data window. No action.
Confidence: HIGH on contents, MEDIUM on the lion/wrong-activation reading.

Other machine-readable "this was bad" markers, all outside the archived window:
`512x19-t81-swa-2446000-badbrokenpath.pb.gz` (2022-10-26) and six
`512x19-t81-swa-*-badrecycle.pb.gz` (2023-01-13/15, coincident with the self-play
Elo dip to 118 right after the 19×512 promotion).

---

## 3. FORMAT SPEC

### 3.1 Container

`.tar` (plain ustar) → members `training.<game_id>.gz` → one gzip stream per member
→ one **complete game**, records in play order, decompressed length an exact
multiple of the record stride. Tars also contain one directory entry (typeflag `5`)
and, in some eras, a plain-text `LICENSE` member that is **not** gzipped — filter on
the `1f 8b` magic, not on the name. Parse in one forward pass; no seeking needed.
Reference implementations: `lczero-training/csrc/loader/chunk_source/tar_chunk_source.cc`
L124–241 and `lczero-training/tf/chunkparser.py` L462–487.

Measured: ~110–140 games per hourly tar sample; 99–125 records per game; 20–50 KB
compressed per game; gzip ratio ~41×.

### 3.2 Record versions

The first 4 bytes are a little-endian version tag. **Dispatch on it**; do not assume.

| ver | tag | stride | `static_assert` source |
|---|---|---|---|
| V3 | 3 | 8276 | `bf5a8d7:src/neural/writer.h:51` |
| V4 | 4 | 8292 | `3eda703:src/neural/writer.h:55` |
| V5 | 5 | 8308 | `1e649cb:src/neural/writer.h:66` |
| **V6** | **6** | **8356** | `src/trainingdata/trainingdata_v6.h:87` |
| V7 | 7 | 8396 | `src/trainingdata/trainingdata_v7.h:46` |

**Everything in ranks 1–8 above is V6/`input_format = 1`**, verified by decoding:
test60 (2021-12), test75, test79, test80 (all five sampled dates), test91 (all
sampled dates). Exceptions found: test60/test71 **2020** tars are V5 (8308);
`training_data/run3/` 2019 tars are V4 (8292). V7 exists in-tree but **nothing on
the server ships it** — it is produced by the new C++ loader's rescorer
(`csrc/loader/stages/chunk_rescorer.cc:42`), not by lc0 self-play, which always
writes `version = 6` (`src/trainingdata/trainingdata.cc:123`).

### 3.3 V6 layout — 8356 bytes, `#pragma pack(1)`, little-endian

Authoritative source:
`https://github.com/LeelaChessZero/lc0/blob/master/src/trainingdata/trainingdata_v6.h`
(L37–88, `static_assert(sizeof(V6TrainingData) == 8356)` at L87).
**I re-derived every offset below by decoding live archive records** — the parse
reproduces the start position, links 709/710 consecutive positions by exactly one
legal move, and yields policy vectors that sum to 1.0000 over the legal moves.

| offset | size | field | notes |
|---:|---:|---|---|
| 0 | 4 | `version` u32 | 6 |
| 4 | 4 | `input_format` u32 | 1 = `INPUT_CLASSICAL_112_PLANE` everywhere in the archive |
| 8 | 7432 | `probabilities[1858]` f32 | **the policy target** — see 3.4 |
| 7440 | 832 | `planes[104]` u64 | 8 history × 13 planes — see 3.5 |
| 8272 | 1 | `castling_us_ooo` u8 | 0/1 |
| 8273 | 1 | `castling_us_oo` u8 | |
| 8274 | 1 | `castling_them_ooo` u8 | |
| 8275 | 1 | `castling_them_oo` u8 | |
| 8276 | 1 | `side_to_move_or_enpassant` u8 | for format 1: **0 = white, 1 = black** (verified: alternates every ply) |
| 8277 | 1 | `rule50_count` u8 | |
| 8278 | 1 | `invariance_info` u8 | b7 stm(canon) · b6 rescorer-delete · **b5 adjudicated** · b4 max-length · b3 best_q proven · b2/1/0 transpose/mirror/flip |
| 8279 | 1 | `dummy` u8 | 0 in V6 (was `int8 result` in ≤V5) |
| 8280 | 4 | `root_q` f32 | **search eval at root**, side-to-move frame |
| 8284 | 4 | `best_q` f32 | search eval of the best child |
| 8288 | 4 | `root_d` f32 | draw probability |
| 8292 | 4 | `best_d` f32 | |
| 8296 | 4 | `root_m` f32 | moves-left, plies |
| 8300 | 4 | `best_m` f32 | |
| 8304 | 4 | `plies_left` f32 | MLH training target |
| 8308 | 4 | `result_q` f32 | **GAME OUTCOME**, exactly ±1 or 0 |
| 8312 | 4 | `result_d` f32 | 1 iff draw |
| 8316 | 4 | `played_q` f32 | eval of the move actually played |
| 8320 | 4 | `played_d` f32 | |
| 8324 | 4 | `played_m` f32 | |
| 8328 | 4 | `orig_q` f32 | **raw NN eval, no search**; **NaN if not cached** |
| 8332 | 4 | `orig_d` f32 | |
| 8336 | 4 | `orig_m` f32 | |
| 8340 | 4 | `visits` u32 | root N *including* root; measured 800–831 (t91), 200–1000+ (t80) |
| 8344 | 2 | `played_idx` u16 | index into `probabilities`; 65535 = none |
| 8346 | 2 | `best_idx` u16 | |
| 8348 | 4 | `policy_kld` f32 | KL(visits ‖ NN policy), nats |
| 8352 | 4 | `q_st` f32 | V7 short-term-Q slot; **always 0.0** in archive data (lc0 does not fill it) |

**Value targets** (`lczero-training/docs/heads.md`):
- game-outcome head: `W = (1 + result_q − result_d)/2`, `D = result_d`, `L = (1 − result_q − result_d)/2`
- **search-eval head ("q"): the same transform on `best_q` / `best_d`** — this is
  the one to use as a teacher score.
- Q → centipawns: for a Stockfish-frame cp use Lc0's own NNUE-export formula
  `cp = 660.6·q / (1 − 0.9751875·q^10)` (`src/trainingdata/rescorer.cc:450`,
  comment: "Formula from PR1477 adjusted for SF PawnValueEg"). The UCI display
  formula is different: `90·tan(1.5637541897·q)` (`src/search/classic/search.cc:322`).

### 3.4 Policy — present, and it is the raw visit distribution

`float probabilities[1858]` at offset 8. It is **N(child)/N(root children)**, not a
softmax and not the NN policy (`src/trainingdata/trainingdata.cc` L137–166):
`fracv = child.GetN() / (float)node->GetChildrenVisits()`.
Illegal / unvisited entries are exactly **−1.0f** (never NaN from lc0; NaN can only
appear after third-party rescoring). Measured on live records: 20 legal entries
summing to 1.000000, 1838 entries at −1.0, zero NaN. Mean max-policy 0.39–0.53
depending on era.

**Move index mapping (1858).** Table: `src/neural/encoder.cc:339` `kMoveStrs[]`,
packed at L574 as `promotion*4096 + from*64 + to` with `kKnight=0,kQueen=1,kRook=2,kBishop=3`
(`src/chess/types.h:52-57`). Squares are **LERF, a1=0…h8=63 — identical to ARCHON's
mapping.** The mirror `lczero-training/tf/policy_index.py` is byte-identical.
Reproduction: (1) for `from` 0..63, `to` 0..63 ascending, emit if queen-move or
knight-move → 1792 entries; (2) for `from` on rank 7 in file order, `to` ∈
{f−1, f, f+1} on rank 8 in that order, suffixes `q`,`r`,`b` → 66 entries. Total 1858.
Indices are in the side-to-move-mirrored frame.

### 3.5 Planes — and the one trap

104 stored planes = 8 history positions × 13. Per history `i`, `base = 13i`:
`+0..+5` = **our** P,N,B,R,Q,K; `+6..+11` = **their** P,N,B,R,Q,K; `+12` = all-ones
iff repetition ≥ 1. `i = 0` is the current position. NN input planes 104–111 are
*not* stored — synthesize them from the scalar fields (`src/trainingdata/reader.cc`
L38–98).

**All 8 history boards are in the side-to-move frame** — vertically mirrored *and*
ours/theirs swapped when black is to move (`ChessBoard::Mirror()`,
`src/chess/board.cc` L57–69; applied per ply at `encoder.cc` L241, L303).

**The trap:** the writer stores `ReverseBitsInBytes(plane.mask)`
(`src/trainingdata/trainingdata.cc:133`) — bits reversed *within each byte*, i.e.
files mirrored within each rank. I hit this empirically before finding the source
line: the start position decoded as `rnbkqbnr`, king and queen swapped. Undo with
the involution:

```rust
fn lerf(v: u64) -> u64 {
    let v = ((v >> 1) & 0x5555_5555_5555_5555) | ((v & 0x5555_5555_5555_5555) << 1);
    let v = ((v >> 2) & 0x3333_3333_3333_3333) | ((v & 0x3333_3333_3333_3333) << 2);
    ((v >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((v & 0x0F0F_0F0F_0F0F_0F0F) << 4)
}
```

**En passant** in `input_format 1` is encoded as a pawn on rank 1 or rank 8 of a
pawn plane — strip that bit and translate it into the ep square.

**Acceptance test for our reader (this is what I used, it works):** decode records
`i` and `i+1` of a game, and assert exactly one legal move from position `i` reaches
position `i+1`. On a test91 tar: **709/710 pass**. Anything below ~99% means the
mirror or the frame is wrong.

### 3.6 Existing Rust readers to crib the shape from (never the code)

- `github.com/jshriver/lc0v6-binpack-converter` — GPL-3.0; `src/record.rs` offset
  constants, `src/fen.rs` plane→FEN incl. ep, `src/binpack.rs` Stockfish binpack out.
  The closest thing to a complete reference.
- `github.com/maxencefrenette/chess-engine-4` — `crates/leela_loader/src/lib.rs`,
  `RECORD_SIZE = 8356`, hand-rolled tar walker, compact policy (indices + f16).
- `github.com/AAce3/dryad-training` — `src/chunk_parser.rs`, `src/policy_names.rs`
  (a second independent 1858 table to diff against).

No crates.io crate exists for this.

### 3.7 What a compact re-encode costs

208 B/position raw. Dropping the 1858-float dense policy is the whole game:
position (32 B packed) + `best_q/best_d/result_q/result_d/plies_left` (10–20 B) +
top-K sparse policy (K=16 → 64 B) lands around **~100 B/position**; value-only
(no policy) lands around **~40 B/position**. So 1 B positions is 40–100 GB on our
disk, from a 208 GB download.

---

## 4. VOLUME ARITHMETIC

| target | GB downloaded | hours @ 25.9 MB/s | cheapest source |
|---|---|---|---|
| 100 M positions | 20.8 | 0.22 | test91, last 5 days |
| **1 B positions** | **208** | **2.2** | test91 20260421–20260801 (419 GB gets 2.0 B) |
| 3 B positions | 624 | 6.7 | test91 2026 windows |
| 6.6 B positions | 1,376 | 14.8 | **all of test91** |
| **10 B positions** | **2,080** | **22.3** | all test91 + test80 back to ~2025-01 |
| 25 B positions | 5,200 | 55.8 | all test91 + all test80 2024-11 → 2025-08 |
| 51.6 B positions | 10,741 | 115 | all test91 + all test80 |
| 135.3 B positions | 28,142 | 302 | the whole archive (not recommended) |

Storage after re-encode, at 100 B/position with sparse policy: 1 B → 100 GB,
10 B → 1.0 TB. Value-only at 40 B/position: 1 B → 40 GB, 10 B → 400 GB.
**Neither 10 B figure fits this box's 126 GB free** — plan for external storage or
a value-only encode before committing past ~2 B positions.

---

## 5. WHAT I COULD NOT ESTABLISH

Ranked by how much it matters.

1. **Whether test91's BT4 net is actually stronger than test80's final 15×768 net —
   in Elo, with a number.** This is the one question that decides ranks 1 vs 2 and I
   did not close it. The training server's Elo columns are **per-run pools with
   reset anchors**: run1 tops out at 205, run2's test91 sits at −26, and run2's
   *legacy* test79 nets sit at 1517 in the same column. Cross-run comparison from
   these pages is meaningless. What I have instead is architectural (BT4-1024x15x32h
   heads Lc0's own "strongest available" list, above the 768x15 shape test80 ended
   on — `https://lczero.org/dev/wiki/best-nets-for-lc0/`) and mechanical (800 visits
   vs a 470 median). Note the weakness in the architectural argument: that page also
   says "the largest network compatible with your hardware is recommended", so its
   ordering is partly by size, and it gives **no Elo figures at all**. Also, test91
   is a *debug* run whose nets may not be a released BT4 iteration.
   **Confidence that test91 ≥ test80 per position: MEDIUM.** If it matters, the
   resolving experiment is cheap and direct: fetch one test91-era net and net
   `826344` from `storage.lczero.org/files/networks/` and play them head-to-head at
   800 nodes.

2. **My SF probe is a NULL for ranking the good eras — say so out loud.**
   test91 / test80 / test75 / test60 all land at 73–79% argmax agreement with SF18
   d13 and 18–29 cp median value error, with n≈250 per cell. The instrument resolves
   *bad* (test79 at 63.3% / 61.8 cp, confirmed at both n=180 and n=400) but it cannot
   separate the top four. **Do not read a ranking off those columns.** Rank 7/8
   (test60, test75) are ranked low on architecture-era grounds and external Elo, not
   on my probe — the probe says they look fine, and that is a statement about the
   probe's ceiling, not about the data.

3. **Why test80's pre-2024-04 data was deleted.** Fact established (listing);
   reason not. **No retention policy is published anywhere** — not in
   `github.com/LeelaChessZero/lczero-server` (the tar packing/pruning is done by an
   out-of-repo script), not on the wiki, not on the blog. The cut is at a clean
   calendar-month boundary and *older* archives (test60 2019, test75 2021, test79
   2022) survive intact, so it was a **targeted purge of the largest directory, not
   an age-based expiry** — ~24 months and plausibly 15–20 TB of T80 data was
   deleted. If the deletion was for a quality reason rather than a storage one,
   there could be a reason to distrust what remains. I found no statement either way.

4. **What test91's seed net actually was.** The Elo curve proves it was seeded from
   something trained (first net −22, vs run3's −4024), and the description says
   "BT4 in RL mode". The anchor row `910000` is recorded with **0 blocks / 0 filters
   and 0 games** — a synthetic row for an imported net, with no sha exposed. So
   "BT4-derived" rests on the run description, not on a hash. There is also **no
   archived training config for T90 or T91** (`training_configs/` stops at 2024-03-17
   for run1 and 2022 for runs 2/3).

5. **Whether any archived window has a known-bad data period the Lc0 team warned
   about.** No public announcement exists for *any* run-level event since mid-2022 —
   the Lc0 wiki's own project history stops at T80 net 801910 and explicitly points
   to Discord `#announcements`/`#dev-log`, which is **not publicly archived**.
   GitHub issues/discussions, the lczero.org blog, Google Groups, Reddit and
   TalkChess were all searched and are dry on T80's termination and on T90/T91's
   launch. The only machine-readable "bad" markers are the `-badbrokenpath` /
   `-badrecycle` net names (Oct 2022, Jan 2023 — both outside the archived window)
   and `test80/broken/`, a directory **created 2026-01-19 and now empty**: someone
   quarantined a batch of test80 data as broken and later deleted it. What was in it
   is unknown (confidence LOW), and it is the single most concerning residual gap,
   because it is the only evidence that part of the archived window may have been
   defective.

6. **Absolute cross-engine ratings for the relevant nets.** CCRL 4040
   (`computerchess.org.uk` and `ccrl.chessdom.com`) was **unreachable** (HTTP 000)
   from this environment, so no CCRL/SPCC per-net table was obtained. The one
   absolute figure found is a single tester's list (Rebel/Ed Schröder, Dec 2025,
   `https://talkchess.com/viewtopic.php?t=85732`): Lc0 0.32.1-BT4 = 3827.5,
   Lc0 0.32-BT3 = 3811.0, BT4 scoring 55.8% over 3000 games vs Stockfish 17.1.
   Confidence MEDIUM (one tester).

7. **The remaining ~5% of my sampling.** Every claim above rests on 3 MB range
   samples (≈14,000 positions each) from 13 tars across 6 eras, plus complete
   directory listings. I did not sample every month of test80, and I did not sample
   test71/test71_5/run3 at all beyond a version check.

---

## Appendix — reproduction

Scratch tooling used for this survey (throwaway, not committed):
`decode.py` (planes→FEN, V6 record parse), `sample.py` (HTTP range → tar walk →
gzip → per-era statistics), `probe.py` / `valerr.py` (SF18 d13 agreement, cp loss,
value error). Range requests work: `accept-ranges: bytes` is honoured, but Cloudflare
can serve a full body on a cache MISS — **always pipe through `head -c N`** or you
will pull a 900 MB tar when you asked for 3 MB (this happened once here).
