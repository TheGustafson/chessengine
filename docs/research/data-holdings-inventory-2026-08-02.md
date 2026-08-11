# Data holdings inventory — 2026-08-02

Taken to size the switch to Lc0 data (`lc0-data-survey-2026-08-01.md`). At the
measured **208 bytes/position**, 1 B Lc0 positions = **208 GB** and 10 B =
**2.08 TB**, against **126 GB free** today. So what we can reclaim decides what
we can hold.

## Filesystem

| | |
|---|---|
| volume | 937 GB total, 764 GB used, **126 GB free (86 %)** |
| `data/` | **684 GB** — essentially all of it is `data/nnue` |

## Breakdown

| dir | size | what it is | fate on switch |
|---|---|---|---|
| `jaxcorpora` | **304 G** | the trainer-format shuffled corpora | **DELETE** — this is precisely what Lc0 data replaces |
| `experiments` | **154 G** | run checkpoints | **SHRINKS** — `ckptreaper` now bounds it (keeps newest 4/prefix, every 100th, KEEP_ALWAYS, never symlinks) |
| `canon-v1` | 83 G | superseded corpus generation (`build`, `build-tight`) | **DELETE** — predecessor of jaxcorpora |
| `fulldist` | 46 G | a `build` tree | **DELETE** after confirming nothing regenerates from it |
| `nets` | 43 G | packed `.cenn` nets incl. `screens/` | **KEEP the canonical + cited nets**, prune the rest |
| `binpacks` | 24 G | SOURCE binpacks (pre-conversion) | **KEEP a sample**; these are the only re-derivable inputs we hold |
| `hardneg` | 15 G | hard-negative corpus + tsv | DELETE (weight 0.000129 in the live mix — effectively unused already) |
| `tmp` | 13 G | scratch packs | **DELETE** — scratch by definition |
| `foldin-tight` | 3.4 G | aux | delete |
| `hardneg-reckless` | 1.6 G | probe | delete |
| `teachers` | 329 M | **the SF teacher artifact** | **KEEP** — regenerable in ~5 s but load-bearing |

## The live mix — 326 GB across 12 corpora

This is what a switch actually replaces:

| corpus | GB | weight |
|---|---|---|
| uho-band75-100 | 81.9 | 0.251 |
| fishpack32 | 51.6 | 0.158 |
| farseerT74-band75-100 | 39.5 | 0.121 |
| datapv2-band75-88 | 31.7 | 0.097 |
| farseerT75-band75-83 | 29.5 | 0.090 |
| multinet-band75-88 | 29.2 | 0.090 |
| t60t70-band75-85 | 26.8 | 0.082 |
| pylon-band75-88 | 15.2 | 0.047 |
| wisr-band75-95 | 11.4 | 0.035 |
| wrongnnue-band75-95 | 8.4 | 0.026 |
| canon-tight-frame | 0.9 | 0.003 |
| hardneg-frame | 0.0 | 0.0001 |

**Note `t60t70-band75-85` (26.8 GB, 8.2 % weight) is ALREADY Lc0 data** — T60/T70
era. So we are not switching to a new source so much as replacing an old,
weak-era sample of it with a certified strong-era one.

## Reclaim arithmetic

| tier | GB | cumulative free |
|---|---|---|
| today | — | 126 |
| + `tmp`, `foldin-tight`, `hardneg-reckless`, `hardneg` | 33 | 159 |
| + `canon-v1`, `fulldist` | 129 | **288** |
| + `jaxcorpora` (only once Lc0 data is proven) | 304 | **592** |
| + prune `nets`, bound `experiments` | ~100 | **~690** |

**~690 GB reachable = ~3.3 B Lc0 positions.** Comfortably past the 1 B floor;
**short of 10 B (2.08 TB) by 3x.** 10 B needs either more disk or a streaming
rescore-and-discard pipeline (lc0's rescorer has `--nnue-plain-file`, so the raw
tars need never land permanently).

## ⚠ Order of operations

**Do NOT delete `jaxcorpora` until an Lc0-trained net has BEATEN the incumbent in
games.** It is 326 GB of the current mix and it is not re-derivable on this box —
regenerating it means re-downloading and re-shuffling the source binpacks. The
cheap tiers (33 GB, then 129 GB) are safe now and buy 288 GB, which is already
enough to bring down and validate a first Lc0 slice.

The safe first move is therefore **`tmp` + `canon-v1` + `fulldist` + `hardneg`
(162 GB)**, keeping every corpus the live mix names.
