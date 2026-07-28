# Canonical engine + net

**This directory is the ONE place the current release net and engine are named.**
Everything else — scripts, docs, tools — must reference these paths and never a bare
filename. Switching the canonical net is then a two-symlink change with no stragglers.

```
engines/canonical/net.cenn   -> the release net        (symlink, target is gitignored)
engines/canonical/archon     -> the release PGO binary (symlink)
engines/canonical/arch.txt   -> cargo features the binary MUST be built with
```

**Use it like this**

```bash
NET=engines/canonical/net.cenn
ENG=engines/canonical/archon
```
```python
from uciprobe import NET          # already resolves to engines/canonical/net.cenn
```

## Current canonical — D48X (gen-2, 1280 accumulator)

| | |
|---|---|
| net | `data/nnue/nets/screens/d48x-1280-sb520-ema.cenn` (138 MB, sb520, heldout 0.006350) |
| arch features | `sota,arch-d48x` — **the binary and net must match**; a mismatched pair fails the CENN section-size check at load |
| binary | `engines/bin/archon-d48x-pgo-v2` (PGO, built by `tools/build_pgo.sh` with the eval-file argument) |
| previous canonical | `d47v-a4b2-cal400.cenn` + `arch-d47v` |

### Elo evidence — recorded honestly

**D48X measured −16.6 Elo [−43.5, +10.2] vs D47V** at the moment of promotion
(n=336, 10+0.1 equal time, both PGO, `engines/runs/d48x-vs-v/`). The confidence interval
includes zero, so it is not proven worse; it is certainly not proven better.

**It was promoted anyway, deliberately, on the owner's reasoning:** D48X is the better
eval, and it loses today because the search is currently too weak to exploit a better
eval at all. This is consistent with the 2026-07-28 loss forensics — on the losses that
are genuinely ours we are *confidently wrong*, discarding the right move early
(`searchopt/LEDGER.md`), which is precisely the regime where a better eval buys nothing.

**Falsifiable prediction this promotion makes:** as search quality improves, D48X's margin
over D47V should *widen*. If search improves measurably and the margin stays flat or
worsens, the premise is wrong and the canonical net should revert. Re-run
`engines/runs/d48x-vs-v` after any landed search gain and record the delta here.

## Loss corpus for this net

`searchopt/positions/` was re-mined for D48X from a **404-game** match vs SF18
(`engines/runs/corpus-d48x`, −132.4 [−157.6, −108.4] Elo, 165 lost games). Corpora are
net-specific and carry a provenance stamp; `classify_losses.py` warns if the corpus was
mined with a different net than the one canonical at the time.

## Changing the canonical net

1. `ln -sfn ../../data/nnue/nets/screens/<new>.cenn engines/canonical/net.cenn`
2. Update `arch.txt` if the architecture changed.
3. `bash tools/build_pgo.sh "$(cat engines/canonical/arch.txt)" <out> engines/canonical/net.cenn`
4. `ln -sfn ../bin/<out> engines/canonical/archon`
5. Record the Elo evidence in this file. **Do not skip this** — a canonical net with no
   recorded evidence is how an unmeasured net sits in production.

The `Bench:` trailer (D24) stays on the **portable default build**, not on PGO binaries,
and is unaffected by any of the above.
