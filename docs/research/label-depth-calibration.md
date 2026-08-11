# Label-depth calibration — is 20k worth 4× the cost of 5k? (2026-08-11)

**Answer: no. Run the relabel campaign at 5k nodes.**

## Method

250,000 records from the head of `uho-band75-100.shuf.bin` (globally
shuffled ⇒ a prefix is a uniform sample; it is also the #1 relabel target at
0.191 mix weight). The SAME slice was relabeled twice with the SAME worker
the cloud campaign uses (`tools/ops/cloud/relabel_worker.py`,
stockfish-eca43a97, CONC=4) at **5,000** and **20,000** nodes. 7,538 rows
(3.0%) dropped as castle-ambiguous by the worker's own rule; the three
score vectors are joined on the 24 position bytes, which the worker leaves
verbatim. Runtime: 202 s at 5k, 777 s at 20k — 16 min total, $0.

## THE INSTRUMENT TRAP (read this before quoting any correlation here)

The first pass read **r = 0.618 between 5k and 20k on identical positions**,
which is absurd for two searches of the same position. Cause: **0.06% of rows
are mate scores at ±30,000** and they own the entire variance. Excluding
`|score| >= 3000` moves the same statistic to **r = 0.9889**. The
`corr_old_new` field written into every campaign `stats.json` — including all
404 fishpack shards — is computed WITHOUT this exclusion and is therefore
compressed by the same effect; do not read it as a label-agreement measure.

## Results (mate rows excluded, n = 242,252)

| pair | r | median \|Δ\| | mean \|Δ\| | sign flip | sign flip (\|x\|>50) |
|---|---|---|---|---|---|
| orig → 5k | 0.9615 | 46 cp | 84.6 | 10.27% | 1.06% |
| orig → 20k | 0.9642 | 44 cp | 77.1 | 10.16% | 0.84% |
| **5k → 20k** | **0.9889** | **15 cp** | **26.8** | 8.06% | **0.21%** |

Label spread (std): original **403.5**, 5k **300.1**, 20k **317.6**.

## Reading

1. **Relabeling is a 3× bigger lever than depth.** Relabeling moves uho's
   labels by a median 46 cp; going from 5k to 20k moves them a further 15 cp.
2. **5k already captures almost all of the correction.** `orig → 5k` (0.9615)
   and `orig → 20k` (0.9642) are nearly identical, while the two relabeled
   versions agree with each other far more closely (0.9889) than either
   agrees with the original. Depth refines; relabeling corrects.
3. **Sign disagreement is concentrated at ~0.** 10% of rows flip sign vs the
   original, but only ~1% among positions the original scored beyond ±50 cp —
   and only 0.21% between 5k and 20k. Near-zero flips are cosmetic.
4. The original labels are visibly flatter/noisier (std 403.5 vs ~300–318),
   which is consistent with uho's high fitted `a = 175.5` against the ~97–109
   that every SF-dev-relabeled corpus lands on.

**Caveat, stated plainly:** this measures label AGREEMENT, not label TRUTH.
20k is presumably the more accurate label; what is measured is that it differs
from 5k by a median 15 cp. Nothing here proves that 15 cp is worthless — only
that it is small, and that no evidence supports paying 4× for it.

## Cost consequence

Measured from the fishpack-v1 campaign's ACTUALS (it ran 1.24× faster than
its plan): **$0.0261 per 1M positions at 5k**, 2.175 vcpu-h/1M, 96-vCPU quota.

| scope | @5k | @20k |
|---|---|---|
| uho-band75-100 (2,560M, 0.191 wt) | **$67 / 2.4 d** | ~$267 / 9.7 d |
| all old-label corpora (8,576M, 0.640 wt) | **$224 / 8.1 d** | ~$896 / 32 d |

**Prerequisite either way:** rebuild the cloud payload — the deployed one
(sha 303fc704…) predates the EOF-clamp fix in `worker_relabel.sh` that the
fishpack campaign's task 403 died on.
