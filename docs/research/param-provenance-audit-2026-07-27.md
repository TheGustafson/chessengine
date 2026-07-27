# Parameter provenance audit — 2026-07-27

Follow-on to the eval-frame study
(`eval-frame-scale-model-2026-07-27.md`), which closed as a negative. That
study rested on a premise worth checking directly: *our cp-denominated
constants sit at Stockfish's raw values.* True — but the study compared
against the wrong Stockfish, and once the right one is used the premise is
stronger than stated and the remaining opportunity moves elsewhere.

Re-runnable: `python3 tools/spsa/param_provenance.py` (exits non-zero on a
canary mismatch or a stale campaign entry).

---

## 1. The reference commit — we were reading the wrong Stockfish, again

`src/search/params.rs`'s header has always named its source: Stockfish
commit **`eca43a97`** (2026-07-06). What nobody checked is where that sits
relative to the tags:

```
sf_16  ──…──  sf_18 (tag, 2026-01-31)  ──223 commits──▶  eca43a97 (2026-07-06)
                    ↑ clone HEAD                              ↑ our real source
```

`eca43a97` is **223 commits *newer* than the `sf_18` tag**, confirmed by
`git merge-base --is-ancestor sf_18 eca43a97`.

The clone sat at `sf_16` until 2026-07-27, when it moved to `sf_18` "to
prevent mistakes" — but for *constants* that swapped one wrong reference for
another. Razoring shows it moving in both directions:

| revision | razoring margin |
|---|---|
| `sf_16` | `alpha − 456 − 252·d²` |
| `sf_18` (tag) | `alpha − 485 − 281·d²` |
| **`eca43a97`** | **`alpha − 465 − 300·d²`** |
| **ARCHON** | `RAZOR_BASE 465`, `RAZOR_DEPTH_MULT 300` — **exact match** |

The consequence for today's work: the frame study reported "razor 465 vs
485" as evidence that our margins are unconverted SF imports. The number is
right and the framing is wrong — 465/300 is not a deviation from anything,
it is a faithful copy of our actual source. Same for ProbCut, where our
`214 − 59·improving` / `428` looked like a drift from sf_18's `235 − 63` /
`418`, and is in fact eca43a97's value to the digit.

**Rule now in CLAUDE.md:** shape questions ("does SF do X?") may use the
checkout; *value* questions must read `eca43a97` explicitly.

---

## 2. The transcription is faithful — no drift Elo exists

Screened all **260 tunables** (238 search + 15 timeman + 7 eval) against
every integer literal in SF's `search.cpp`, `movepick.cpp`, `evaluate.cpp`,
`timeman.cpp` and the three headers at `eca43a97`. Values below 20 are
skipped (they collide by accident).

**Exactly two values ≥20 appear nowhere in SF, and both are expected:**

| tunable | value | why |
|---|---|---|
| `RFP_THREAT_MARGIN` | 55 | ARCHON original — Wave-G item 23 (D30), threat-coupled RFP relaxation |
| `INFO_THROTTLE_NODES` | 10 000 000 | UCI plumbing, not a search constant |

In-place spot checks (the screen only proves a value *exists* in SF, not
that it sits in the right formula) all matched exactly:

- razoring `465 / 300·d²`
- ProbCut `beta + 214 − 59·improving`, TT-ProbCut `beta + 428`
- the entire movepick scoring family — `7·PieceValue` capture MVV, `2·main`
  + `2·pawn` quiet weights, the continuation set **[0,1,2,3,5]** (note: not
  [0..4] — SF skips offset 5 and reads offset 6), `16384` check bonus gated
  by `see_ge(m, −75)`, `20·PieceValue` threat escape/entry, `8·lowPly/(1+ply)`,
  `1<<28` evasion capture bonus, `goodQuietThreshold −14000`, the
  `see_ge(m, −score/18)` good-capture split, `−3560·depth` partial sort

**Conclusion: there is no transcription-drift Elo to harvest.** Combined
with the frame study, the "our constants are wrong" hypothesis family is now
closed on three independent fronts:

1. margins scaled DOWN by the measured frame ratio — **refuted** (negative)
2. eval scaled UP by its inverse, the dual, which also repairs class 4 —
   **refuted** (negative)
3. the constants are mis-transcribed or stale relative to SF — **refuted**
   (this audit: faithful to the digit, from a 2026-07 master)

The values are right, they came from a recent master, and moving them as a
group loses Elo. Remaining search Elo is therefore **not in the constant
values.**

---

## 3. Where the audit says to look instead

### 3a. ARCHON originals — the only constants with no external tuning behind them

Two tunables carry no Stockfish provenance at all, so no SF tuning run has
ever validated them, and (see 3b) no SPSA campaign of ours has either:

| tunable | value | what it does |
|---|---|---|
| `QUIET_OFFENSE_FACTOR` | 10 | Wave-G offense term (D30): a quiet moving to a square from which it attacks an *undefended* enemy worth ≥ the mover earns `10·piece_value` in the ordering score. The "create a threat" counterpart to SF's escape-only term. |
| `RFP_THREAT_MARGIN` | 55 | Wave-G item 23 (D30): relaxes the RFP margin when the threat map says nothing of ours is attacked. Enters its margin with a **minus** sign — shrinking it *enlarges* the margin (misread once today). |

Both sit in hot paths (move ordering; the RFP gate), both were introduced by
reasoning rather than measurement, and both are unvalidated. These are the
natural first ablations — and unlike a frame rescale, an ablation here has a
clean null (set to 0 = remove the term).

### 3b. SPSA coverage: 121 of 260 tunables (46.5%) are in no campaign

Not automatically suspect — many are unit-defining denominators that are
deliberately pinned (`RED_UNIT`, `ASPIRATION_WIDEN_DEN`, `CORR_UPDATE_DEN`,
the `*_DEN` family). But whole *mechanisms* have never been tuned:

| family | uncovered | note |
|---|---|---|
| movepick scoring | 9 | verified faithful above, so low risk — except `QUIET_OFFENSE_FACTOR` |
| ProbCut | 5 | `PROBCUT_BETA_BASE/IMPROVING/MIN_DEPTH/DEPTH_SUB/CHECK_BETA` |
| LMR finalization | ~14 | `RED_TTPV_*`, `RED_CUTOFFCNT_*`, `RED_ALLNODE_*`, `RED_TTCAPTURE` |
| TT-cutoff verification | 6 | `TTCUT_*`, `TT_*` |
| fail-low / `FL_*` | 15 | never in any campaign |

### 3c. One stale campaign entry

`s4-searchgap.json` lists `CHECK_EXT_MIN_DEPTH`, which exists only on the
unmerged `wip/check-ext` branch. The campaign's own comment says so
("param present only there"), so it is knowingly forward-looking rather than
a bug — but it will fail an SPSA preflight against a `main` binary. The
audit tool flags it.

---

## 4. What this changes about the plan

The standing conclusion from
`where-the-sf16-gap-actually-is-2026-07-27.md` — that the −78 Elo vs SF16
splits ~−33 eval quality and ~−45 speed — is unaffected and now better
supported: with margin calibration refuted *and* transcription verified,
the −33 equal-node component cannot be explained by our constants at all.

The tunables work that remains worth doing, in order:

1. **Ablate the two ARCHON originals** (§3a) — cheap, clean null, genuinely
   untested. Needs an idle box.
2. **Shape diff, not value diff** — enumerate mechanisms present in
   SF@`eca43a97`'s `search.cpp` that we do not implement. If the constants
   are right and the tree is already smaller than SF's, missing *shapes* are
   where search Elo is left.
3. **ProbCut / LMR-finalization SPSA** (§3b) — the largest untuned
   mechanisms, deferred until a box is free.

Item 2 is the one that does not need compute, so it is next.
