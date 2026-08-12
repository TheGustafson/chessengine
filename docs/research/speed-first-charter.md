# Charter reset: the engine is now a SPEED engine (owner, 2026-08-11)

**The goal is NPS maxing — the fastest possible search.** Features that cost
speed go, including ones we built and liked. The eval is a small, fast net
(`arch-hka128`: HalfKA-only, 128 accumulator, 16/16 head, 18 queen-split
buckets); the search must be retuned for it, because every tunable we ship was
transcribed from Stockfish and assumes a **bigger, more accurate, more stable**
evaluation than ours now is.

This supersedes the D41 learned-search program and the value-foundation
"beat SF16 at equal time by out-evaluating it" framing.

---

## 1. What is being cut, and the honest reason for each

| feature | cost | evidence |
|---|---|---|
| **Learned policy head** (`policy_head.rs`, `policy_vocab.rs`, `.cpol` sidecar, `pol_lambda`) | FT-adjacent work + a whole sidecar container | **Tested and failed its bar.** `pol_lambda` was set to 0 and the D41 policy era ditched after top-1 accuracy never reached the owner's 35 % bar. Already inert; now it leaves the tree. |
| **Disagreement head** (`DIS1` section, `dis_*` tensors, DIS-TM, DIS-census) | a second head, a polled forward pass, search-side plumbing | **Never resolved.** Its one arm read **+9.38 ± 10.64 (LOS 95.8 %)** and was never confirmed at power. Cut for speed and simplicity, not because it was refuted — recorded honestly so nobody later "remembers" it as a failure. |
| **FullThreats** | 60,720 of 83,248 input rows; the FT working set falls 17.3 MB → 2.88 MB without it | The single biggest speed lever we have. Deletion planned after a working 128 net exists (so the removal has a live acceptance test). |
| **Learned search allocation** (design/19: `search_trace.rs`, `search_intervention.rs`, obs store, consequence heads) | per-node observation plumbing | The premise was that a net could learn to allocate search. At NNUE sizes that stayed unproven, and the machinery is pure overhead on a speed engine. |
| **Variance / WDL heads** | extra tail weights + forward work | No shipped consumer. |

**Correction to the framing, stated once:** "proved not possible" is right for
policy (measured against a stated bar and missed) but stronger than the
evidence for disagreement (never powered either way). The decision to cut both
is sound on speed grounds regardless; the record should just not claim a
refutation that was never run.

---

## 2. What replaces them as the program

### 2a. NPS is the metric

Every change is judged by nodes per second at equal strength-per-node, and the
existing measurement discipline carries over unchanged — `instructions:u` and
node counts are the only counters that survive their own null (the counter
trust table in CLAUDE.md), and the byte-identical null still gates any panel.

Known state of the speed frontier, from the perf studies already banked:
- **The FT fold is the wall**, and it is MSHR-limited, not proximity-limited —
  locality, row reordering, prefetch and sub-byte weights are all CLOSED by
  construction. `arch-hka128` attacks it the only way left: make the working
  set small enough to stop being a memory problem (2.88 MB).
- **Code layout is closed** at every granularity (study 68).
- **Branch mispredicts are ~6 % of cycles with no hot spot**; the break-even
  for a branchless rewrite is 22.4 instructions per mispredict removed.
- With the trunk shrunk, **the head stops being negligible**: at L1=128 the
  L1→L2 stage is comparable to the FT work per node, where at 768 it was noise.
  Head-side cost now deserves the attention the FT used to get.

### 2b. Retuning the search is now first-class work, not cleanup

**223 tunables, 60 ever touched by an SPSA campaign, 163 running on faith** —
and all of them transcribed from a Stockfish whose eval is bigger, more
accurate and more stable than ours. Margins, reductions and pruning thresholds
that are correct for SF's eval are not correct for a 128-wide HalfKA-only net;
they were never ours to begin with.

The program already exists in `deviation-program.md` and is promoted here:

1. **Simplification ladder** (S1 singular collapse, S2 SF-cargo, …) — each
   batch removes tunables and gets a non-inferiority SPRT. Fewer knobs is both
   a speed win and a prerequisite for tuning the rest.
2. **Stage-B SPSA** (S7) — retune the survivors on OUR games with the new net.
   This is what ends the transcription lineage: `param_provenance.py` flips
   from a faithfulness check into a divergence certificate.

Sequencing note: tune AFTER the net shape settles. Tuning against an eval we
are about to replace wastes the campaign.

---

## 3. Rename

The name should say what the engine is for. ARCHON was chosen for a
"Stockfish-class architecture" ambition that this charter retires. Candidates,
owner's call — the repo/crate rename is mechanical (Cargo `name`, UCI `id
name`, docs) and should happen in ONE commit, before any public release:

| name | why |
|---|---|
| **TEMPO** | a chess term for a gained move AND a word for speed — says the thesis in one word |
| **CELERITY** | swiftness; distinctive, no collisions in the engine list |
| **QUICKSILVER** | fast and light; memorable |
| **DART** / **BOLT** | short, blunt, obviously about speed |

---

## 4. Order of work

1. Finish the `arch-hka128` smoke: spec → train → export → netpack → engine
   load → eval. (in flight)
2. First real small net; confirm it loads and plays.
3. **Delete FullThreats** — with the new net as the acceptance test.
4. Delete policy, dis, obs/trace/intervention, variance/WDL heads.
5. Measure NPS honestly against the old canonical (cross-binary matches
   already support this: `cross_arch_screen.sh` takes CAND_BIN and REF_BIN
   separately, so the frozen PGO binary + canonical net stays the reference).
6. Simplification ladder, then stage-B SPSA on the new eval.
7. Rename, license, release prep.

---

## 5. The frame problem, measured (2026-08-11, T1 @ sb32)

The first real scale check on the fast arch, against canonical:

| check | median | band | verdict |
|---|---|---|---|
| relative to canonical | **0.796** (IQR 0.66–1.01, slope 0.751) | 0.85–1.18 | FAIL |
| absolute vs the SF internal frame | **0.445** (slope 0.373) | 0.80–1.25 | FAIL |

**sign-agree 97.4 %** — the net is learning real evaluations; only the
MAGNITUDE is wrong. In this net's units the transcribed cp constants are
**2.68× too large**, which is the owner's "the tunables are meant for a
bigger, more accurate, more stable NNUE" with a number attached.

Three things follow:

1. **`network_scale` is a pack-time HEADER field, not a training property.**
   The 600→ promotion precedent (`859a442`) changed four header bytes with the
   payload byte-identical. So the frame can be corrected on any checkpoint
   without retraining — cheap, and it should be re-derived once the net is
   converged, NOT now at sb32 where the magnitude is still growing.
2. **Do not chase SF parity.** 6000 games established that the shipped frame
   IS the optimum and that SF parity costs ~18 Elo
   (`eval-frame-0538-sf-internal`). The target is CANONICAL's frame
   (the 0.796 number), not the absolute 0.445 one.
3. **The scale check is ADVISORY in the gate script** — it prints and the
   match runs anyway. So a mis-scaled net produces a real-looking Elo number
   that is actually a units artifact. Any T1 rung must be read with the scale
   line next to it, or it will be misattributed to eval quality.

Re-derive at sb100+ with `tools/nnue/calibrate_network_scale.sh`, then pack
the gate nets with the corrected scale.
