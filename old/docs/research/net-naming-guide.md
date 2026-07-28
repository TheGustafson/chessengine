# Net naming guide

Names grew a segment at a time as campaigns evolved; this decodes them.
The canonical id is the ledger row id (`ledger/nets.jsonl`); the file is
always `data/nnue/nets/<id>.cenn`.

## The general shape

```
<campaign>-<gen?><width>-<data?>-<schedule>sb
stage2-v1 - s3    2048  - farseer - 120sb
```

Read right to left: how long it trained, on what data, at what width,
under which campaign.

## Segment by segment

**`…sb` — schedule length.** Superbatches trained. 1 sb ≈ 100M position
presentations in bullet, so `240sb` ≈ 24B presentations (with repeats over
the corpus). Longer ≠ better by default — it interacts with the LR
schedule, which is why 120sb and 240sb versions of the same thing both
exist.

**Width — `256 / 512 / 1024 / 2048 / 4096`.** The hidden-layer size of the
classic arch `768→2×W→1`. Sometimes prefixed:
- `u256`, `u1024` … = stage-1 net trained on the **UHO 10GiB head** sample
  (`uho-ladder-v1` era).
- `uf512`, `uf2048` … = **UHO full** 40GB binpack (`uho-full-v1` and
  `uho-wave*` eras). The `f` is the important bit: full data.
- bare `1024`/`2048` inside `stage2-v1-*` names = width of the warm-started
  net (its parent is the matching `uf*` stage-1 net).

**`s2 / s3 / s4` — stage-2 generation.** Counts warm-retrain hops off the
stage-1 base: `s2` = first stage-2 attempt (test80 labels), `s3` = second
(farseer T76 labels — the current INCUMBENT), `s4` = third (farseer T74).
Not "version numbers" — each is a different corpus experiment.

**Data segment** — the stage-2 label corpus: `test80` (lc0 test80 dump,
weak labels, mostly dead end), `farseer` (farseer T76 binpack — the good
one), `t74` (farseer T74).

**Campaign prefixes.**
- `net-0 … net-10` — the first era, hand-numbered. `net-0` (sf-100 data,
  256-wide) was the champion for weeks; almost everything after it is
  measured against it. net-1..6 = the lc0-label failures (label-scale bug
  era), net-8..10 = 512-wide variants.
- `uho-ladder-v1` — width/schedule ladder on the 10GiB UHO head.
- `uho-full-v1`, `uho-wave3/4/5` — successive stage-1 campaigns on full
  UHO (wave3 = schedule probes, wave4 = 2048 width, wave5 = 4096 width).
- `stage2-v1` — the stage-2 warm-retrain program on top of those.
- `d40-*` — the FULL-FEATURE arch (D40: SF HalfKAv2_hm + FullThreats +
  Reckless-shaped head), not the classic 768 feature set. `d40-uho-v1-240sb`
  = stage-1 on full UHO; `d40-uho-720sb` = same but 720 superbatches;
  `d40-s2-farseer-120sb` = its stage-2 hop.
- `e1 / e3 / …` — one-off controlled experiments; `classic512-net0-expand`
  = net-0 upscaled; `*-QUAR`/`DELETED` markers = quarantined/removed data
  lineage.

## The four names that matter today

| Name | Role |
|---|---|
| `stage2-v1-s3-2048-farseer-120sb` | **INCUMBENT** shipped evaluator ("s3") |
| `stage2-v1-s4-2048-t74-240sb` | working-best ("s4"), TC re-check pending (promotion #5) |
| `d40-uho-v1-240sb` | best parity-passing D40 (lost gate #4 on speed) |
| `net-0` | historical baseline all gains compose from |

Log shorthands `s3`, `s4`, `u1024`, `uf2048` always refer to these full
ids; the h2h/gate RESULT lines in `chain-v2-results.log` use the short
forms.

## Convention going forward (D45 era)

New nets keep `<track>-<width>-<data>-<schedule>sb`, e.g. a hypothetical
`d40r-2048-uho-240sb`. Since the evaluator is frozen, expect few new net
names until the D41 shift; anything new gets a ledger row at creation,
which is the authoritative decoder.
