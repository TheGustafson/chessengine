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
over D47V should *widen*. Re-run `engines/runs/d48x-vs-v` after any landed search gain and
record the delta here.

**This is a HUMAN decision and is not delegated.** Net choice, architecture, accumulator
width and retraining are decided outside `searchopt/`; an agent doing search work must not
spend time reasoning about them. Recorded here for context only.

**If the prediction fails, DO NOT REVERT TO D47V** (owner ruling, 2026-07-28). The gen-2
direction stands; what would be wrong in that case is the width, not the recipe. The
response is to **retrain at a smaller accumulator** — the NPS cost of 1280 is real and
measured (907 vs D47V's 1192 knps, 0.76x), so if the search cannot convert the better eval
into Elo, the fix is a net that costs less per node, not a rollback to the old generation.
D47V remains on disk as a comparison yardstick only.

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

---

## 2026-07-28 (evening) — engine binary updated to the merged perf work

`archon` now points at **`archon-d48x-s10pgo`**, a PGO build of `main` after perf
studies 8 and 9 were merged (`274f0ad`). The net is unchanged — this is a **speed and
quantisation change only**, not a new generation.

| | before | after |
|---|---|---|
| binary | `archon-d48x-pgo` | **`archon-d48x-s10pgo`** |
| instructions / node | 7,116.6 | **6,429.9 (−9.65%)** |
| ratio vs SF18 | 1.644× | **1.4851×** |
| NPS vs SF18 | — | **0.7014** (measured on an idle box, study 9) |

**Verified through the pointer:** `bench 16 1 13 … depth net.cenn` = **1,922,947**, and
`CHECK_NNUE_GATE=1 bash tools/check.sh` exits 0 with eval hash `5ca978e0…8b38` and
`Bench: 587683`.

**The eval is NOT bit-identical to the previous canonical.** Studies 8 and 9 landed two
quantisation changes (FT frame ratio carried as 4 rather than 255/64; the output-skip lane
as an integer `vpdpbusd` dot) measured at **0.14 and 0.94 SF cp mean** against an 800-position
fidelity panel. Both were adopted on a *fidelity* gate rather than a bit-exactness gate —
that distinction is what unblocked them, and the same gate rejected a third variant at
max 205 SF cp.

**Strength is not yet independently confirmed.** The mechanism is: **+0.099 ± 0.226 ply at
equal time**, paired within games (n=4,479 moves), against **+0.13** predicted from the
instruction saving — prediction and measurement agree. At 2.02 ply/doubling (measured in
games) that is worth **+3.5 to +6.0 Elo** depending on the Elo/doubling constant, whose
95% CI includes zero. An adoption SPRT (`elo0=-5 elo1=0`) was paused at n=117 for memory
pressure and should be resumed.

---

## 2026-07-30 — network_scale 400 → 600 (same weights, header only)

**The net file changed by four bytes and nothing else.** `net.cenn` now points
at `d48x-1280-sb520-ema-s600.cenn`, which is `d48x-1280-sb520-ema.cenn` with
`network_scale` patched from 400 to 600. Payload is **byte-identical from offset
64** (`cmp -i 64` → identical). The weights, architecture, and `arch.txt`
(`sota,arch-d48x`) are unchanged, so the existing engine binary loads it without
a rebuild.

### Evidence

| screen | result |
|---|---:|
| @600 vs @400, n=1466 | +15.65 +/- 11.02 |
| **@600 vs @400, n=5880 (confirmation)** | **+14.78 +/- 5.45** (exp-0167) |
| @900 vs @600, n=1466 | −28.50 +/- 10.97 |

Fixed 25,000 nodes, `UHO_Lichess_4852_v1.epd`, both sides the same binary with
only `EvalFile` differing. The optimum is bracketed inside (400, 900) near 600;
**over-scaling costs more than under-scaling gains**, so 600 is taken rather
than pushed higher.

### Why 400 was wrong

400 is netpack's **default**, never a tuned value — the same default D49 also
ships. It was never swept for any net. The optimum is **net-specific**: D49
prefers ~756 (eval stdev 415) while D48X prefers ~600 (stdev 610), and the
hypothesis that a fixed ~610 stdev target would transfer was **refuted** — D49
at 1111 (stdev 600) lost −14.47 +/- 11.09 to D49 at 756.

### Mechanism, corroborated by the pin move

Our pruning margins are hardcoded transcriptions of Stockfish constants
(razoring `alpha − 465 − 300·d²`, futility, ProbCut) and are calibrated against
an eval of a particular spread. Correcting the spread makes them bite correctly
— and the **d13 node total falls 1985398 → 1739588, −12.4%**. The better
calibrated net reaches depth 13 on 12.4% fewer nodes, which is an efficiency
gain rather than a scoring artifact.

### Pins moved in the same commit

| pin | old | new |
|---|---|---|
| `Bench:` | 587683 | **587683 (unchanged)** — a default-feature PSQT run, blind to the net |
| NNUE eval hash | `eb981b11…c84f4f37` | **`2e1ede95…93cf79c4`** |
| d13 nodes | 1985398 | **1739588** |

### Reverting

Point `engines/canonical/net.cenn` back at `d48x-1280-sb520-ema.cenn` and
restore the two NNUE pins. The original file was never modified.

## PROVENANCE GAP — record the build commit (added 2026-07-31)

**`engines/canonical/` records no commit for the binary it ships.** On
2026-07-31 that made a real question unanswerable.

**The question.** Perf study 60 found that `build_pgo.sh` had been silently
emitting **untrained** PGO binaries: `engines/canonical/net.cenn` is a symlink
into gitignored `data/`, so **it dangles in every fresh git worktree** — where
every perf study since 40 has run — and the training session failed its
`setoption` without stopping the build. Measured, node-identical:

| build | i/node |
|---|---:|
| non-PGO native | 5819.68 |
| PGO trained **with** the net | **5728.98** |
| PGO trained **without** the net | **6040.90** |

A netless PGO binary is **5.445% worse than the correct one and worse than no
PGO at all**. `build_pgo.sh` now refuses to emit one (`[ -r ]` on the eval-file,
plus a grep for `ERROR EvalFile` in the training log).

**Why it could not be answered for `archon-s25`.** It is the default engine for
five tools (`screen.sh`, `paired_nps.py`, `frame_compare.py`, `eval_bench.py`,
`reprgap/twins.py`), so a 5.4% degradation would matter — most of all to
`paired_nps.py`, which measures speed directly. Study 60's discriminator is
unambiguous (correct PGO −1.559% vs non-PGO native; netless +3.801%) but it needs
**a build at the binary's own commit**, and that commit is not recorded:

- `uci` reports only `id name ARCHON 0.1.0` — no build identity.
- This file carried no provenance line.
- A structural test (are the NNUE kernels in a cold text section, as they would be
  if never executed during profiling?) **does not discriminate** — both the
  canonical binary and a known-good PGO build expose a single `.text`.

**Requirement going forward.** Any binary placed in `engines/canonical/` records,
here, in the same commit that promotes it:

1. the **git commit** it was built from,
2. the **exact build command**, including the `--features` string and the
   **resolved absolute path** of the eval-file (never the `net.cenn` symlink —
   that is what dangles), and
3. the binary's **`.text` md5**, so the artifact can be tied back to a rebuild.

Provenance by artifact — mtime, filename, or "I remember building it" — is not
provenance. The same failure produced a phantom −0.473% measurement on
2026-07-31 that a rebuild corrected to −3.056%.
