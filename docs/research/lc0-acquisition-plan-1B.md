# Lc0 acquisition — first 1 B positions

Plan of record for the first slice. Survey and evidence:
`lc0-data-survey-2026-08-01.md`. Inventory: `data-holdings-inventory-2026-08-02.md`.

## What we are taking, and why

**The newest ~694 hourly tars of `test91`** — i.e. the most recent ~29 days of
Lc0's only *active* run.

| | |
|---|---|
| source | `https://storage.lczero.org/files/training_data/test91/` |
| filename | `training-run2-test91-YYYYMMDD-HHMM.tar` |
| generating net | **15×1024, BT4-class** (run2 nets `910830`–`913222`) |
| visits per recorded position | **fixed ~800** (measured p1=800, p99=831) |
| policy | **yes** — full 1858-float MCTS visit distribution |
| format | **V6**, 8356 B/record, packed |

**Why test91 and not test80 for the FIRST slice**, despite test80 being 7× larger:

1. It is the **only active run**, on the **BT4** architecture, which heads Lc0's
   own best-nets list above BT3 and T82.
2. It was **seeded from a trained net, not from scratch** — its first net rates
   −22 where a genuine from-scratch run a month earlier starts at **−4024**. This
   is the specific hazard the owner raised (data from a net that is still
   learning), and it is the one thing that most needed establishing.
3. Its search is **deeper per recorded position**: fixed ~800 visits vs test80's
   ~470 median.
4. It is the **most recent** data, which the owner asked for.

**Honest caveat carried from the survey:** test91-vs-test80 net Elo is **NOT
closed** — the training server's Elo pools are per-run with reset anchors, so
test80's 205 and test91's −26 are not comparable. Confidence MEDIUM. If the first
slice underperforms, test80 is the fallback and it is 45 B positions deep.

## Arithmetic

| | |
|---|---|
| measured | **208 B/position**, **25.9 MB/s** from this box |
| tar size | ~299.7 MB avg (1,375.7 GB / 4,590 tars) |
| positions per tar | ~1.44 M (6.61 B / 4,590) |
| **for 1 B positions** | **~694 tars = 208 GB = ~2.2 h** |
| disk free now | **281 GB** (after the 155 GB reclaim) |

## ⚠ Stream-convert-and-discard, NOT download-then-convert

208 GB of tars into 281 GB free leaves 73 GB — not enough to also hold the
converted corpus. So the pipeline must be incremental:

```
for each tar:  download -> convert -> append to corpus -> DELETE the tar
```

Peak disk = converted corpus + one tar, not both in full. This also means a
failure costs one tar, not the whole slice.

## Manifest — so we know what to get next

Every downloaded tar is recorded in `data/nnue/lc0/manifest.tsv`:

```
tar_name	bytes	sha256	positions	converted_at	status
```

**This is the whole point of the exercise**: the next slice continues *backwards*
from the oldest tar in the manifest, with no overlap and no guesswork. Without
it, a second download either re-fetches what we have or leaves a hole.

## What to get later, in order

| next | what | positions | GB |
|---|---|---|---|
| 2 | rest of `test91`, walking backwards | 5.6 B | 1,168 |
| 3–7 | `test80`, 5 windows newest-first | 45.0 B | 9,365 |
| — | `test60`, `test75` — older architectures, lower priority | 57 B | 11,854 |
| **never** | `test79` (defective, 63.3 % SF agreement), `test78`/`test90` (empty), 2018–2020 dirs | | |

## ⚠ OPEN: is test91 actually the best source? (owner asked 2026-08-02)

**Not established.** The survey's own SF-agreement probe is a NULL for ranking the
good eras — test91/test80/test75/test60 all land at 73-79 % argmax agreement at
n~250. It resolves BAD data (test79, 63.3 %) but cannot separate the top four.
The test91 ranking rests on external evidence at stated MEDIUM confidence.

| for test91 | for test80 |
|---|---|
| only ACTIVE run; BT4 arch heads Lc0's best-nets list | **BT3 AND BT4 were both trained on test80** — proven to produce a BT4-class net |
| seeded, not from scratch (-22 vs -4024 cold start) | **45 B positions vs 6.6 B** — 7x deeper, and we want 10 B |
| **~800 visits/position vs test80's ~470 median** — deeper search, better labels | test91 is described as a "**debug** run for training BT4 in RL mode" |
| most recent | — |

**Neither side is decisive.** test91 has the deeper per-position search; test80 has
the proven downstream result and the volume. The Elo comparison is NOT closed
(per-run pools with reset anchors).

**RECOMMENDED: hedge the first slice** — take roughly half from each and let our
own training arbitrate. That is the only instrument that measures what we care
about, and it costs one extra download instead of a wrong commitment to 10 B of
the worse source.

## Open items before the data is trainable

1. **Target transform must be re-derived.** Our per-corpus `(k, p)` frame fitting
   assumes SF-style eval labels. Lc0 gives game outcome + MCTS Q, a different
   space. The existing `(k, p)` values do NOT transfer.
2. **Reader must handle the byte-reversal trap.** Plane bits are stored
   byte-reversed; the survey hit it empirically (start position decoded as
   `rnbkqbnr`) before finding `ReverseBitsInBytes` in the lc0 source.
   **Acceptance test: consecutive records must link by exactly one legal move**
   (survey measured 709/710 on a test91 tar).
3. **Consider lc0's own rescorer.** The archive is RAW self-play; lc0's rescorer
   does tablebase rescoring and `--deblunder` value repair, and has an
   `--nnue-plain-file` export path. That may be a better ingestion route than
   writing our own reader, and it addresses quality at the same time.
4. **`test80/broken/`** was created 2026-01-19 and is now empty — someone
   quarantined part of that window as broken and deleted it. Unexplained.
   Matters only when we reach test80.
