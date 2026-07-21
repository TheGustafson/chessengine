# The scalar-value data contract (D46, 2026-07-21)

The one-page operational contract for EVERY pipeline that produces, transforms,
or consumes scalar-value NNUE training data. Written after the canon −582
post-mortem (`sf18-canon-data-diagnosis-2026-07-21.md`) proved that both 2026-07
corpus poisonings came from the same class of silent recipe drift — not from bad
labels. The governing decision is **D46** (design/decisions.md); the platform
doc is design/16 §3.10. Code enforces this contract by default; deviating
requires an explicit, loud opt-out.

## 1. The five filter criteria (trainer parity)

A scalar value loss may only see positions that pass ALL of
(`sfbinpack_filter`, `tools/nnue/bullet/d40_features.rs` — the normative copy):

1. `ply >= 16`
2. side to move NOT in check
3. `|raw score| <= 10000` — tested on the RAW score, BEFORE any clamp (rule §3)
4. best move has kind Normal (no promotion / castling / en-passant)
5. best-move destination empty (non-capture)

Rationale: a static evaluator cannot represent a label that prices a tactical
resolution (check evasions, capture sequences, mate lines); such labels are
irreducible high-magnitude noise exactly where search stands pat. Measured
poison rates when skipped: canon 27.5%, hard-negs 32% (incl. 4.5% mate-scale).

**Where the filter runs — at the layer that still has raw scores + best moves:**

| Data path | Filter site | Mechanism |
|---|---|---|
| sfbinpack streaming (binpack → trainer) | trainer load time | `sfbinpack_filter` (automatic) |
| .ceobs shards → text corpus | `obs-tool export-value` | **ON BY DEFAULT**; `--no-filter` = audit-only, warns |
| binpack → TSV extraction | `fast_fens` | mode argument REQUIRED: `filter` \| `nofilter` |
| FEN relabel (python) | `relabel_depth.py --curate` | curate = training-grade; non-curate = pilot instrument, never train on it |
| policy corpus (`export-train`) | **deliberately unfiltered** | policy heads must see tactical positions; this contract is for the SCALAR VALUE loss only |

## 2. The scale law (never guess a multiplier)

`engine value = l3 · network_scale`, netpack bakes `network_scale = 400` for
every recipe, and training gives `l3 ≈ training_score · SCORE_MULT / 400`. So:

```
engine value ≈ training_score · SCORE_MULT
```

| Label source | training_score units | SCORE_MULT for incumbent-scale output |
|---|---|---|
| official binpack via sfbinpack loader | internal/1.9 (SFBINPACK_SCORE_DIV) | 1.0 |
| SF18 UCI cp (canon/hardneg/relabel), WDL=0.3 | raw UCI cp | **1.42** (= K/1.9, K = binpack/UCI = 2.64–2.76 measured; **validated in-dist 1.004**, exp-0124) |
| SF18 UCI cp, WDL=0 | raw UCI cp | **1.94** (empirical output-match, exp-0111; **validated in-dist 1.022**, exp-0125) |
| `D40_MIX_DATA` aux beside an sfbinpack base | MUST be pre-scaled to binpack/1.9 units **at build time** (raw-UCI cp × 1.42, clamp ±2499) | 1.0 (the global MULT is shared — the trainer applies NO per-source scaling, so a raw-UCI aux would run at 0.51× its own base; this also invalidated exp-0118 a third way) |

- **MULT is coupled to WDL** (the blend sharpens effective output scale);
  `train_canon.sh` auto-couples. Never copy a MULT between WDL settings.
- The false comment "SF-UCI cp == binpack/1.9" caused the −582; it is dead.
  Any new scale claim must be MEASURED (the K probe method is in the diagnosis).
- **Enforcement:** `tools/nnue/net_scale_check.sh` (band 0.85–1.18, sign ≥95%)
  runs as a HARD abort in every gate script before any games. Every newly
  trained net passes it before matches; FORCE=1 exists only for deliberate
  calibration experiments.

## 3. The clamp-order rule

**Filter first, clamp last.** A ±2500 clamp applied before the |score|≤10000
criterion silently converts mate labels into plausible-looking evals (86,320 of
them sat in the hardneg aux). Any tool that must clamp for a format invariant
(bf-convert's |score|<2500) may only do so AFTER the filter has seen the raw
score. `export-value` implements this internally; `relabel_depth.py` non-curate
output clamps without filtering and is therefore pilot-only by contract.

## 4. WDL (result) column

- A scalar corpus carries either REAL White-relative game results (join them:
  `build_canon_tight.sh` / `build_hardneg_corpus.sh` / cloud builder patterns)
  or trains with `WDL=0` + the matching MULT. A 0.5-placeholder column trained
  with WDL>0 is corruption; a real column trained at WDL=0 wastes signal.
- Builder self-check: result distribution must not be degenerate (>70% one
  value ⇒ die). Reference: UHO extraction ≈ W42/D47/L11 White-relative.

## 5. Builder self-checks (every corpus builder dies loudly on drift)

`build_canon_tight.sh` is the template: (a) filter drop-rate inside an expected
band (20–35% for SF-game distributions — a rate of ~0% means the filter didn't
run; ~100% means a field is broken); (b) WDL join hit-rate against an explicit
JOIN_MIN with the reason for any relaxation documented in-script; (c) result
distribution non-degenerate; (d) .bin 32-byte alignment + text/bin row match;
(e) every corpus gets a D38 ledger row with the drop/join numbers in its recipe.

## 6. Bound quarantine (standing, D38-era)

Only `Bound::Exact` + `Completion::Complete` references may become scalar
labels (`export-value` enforces). Aspiration-window bounds and timed-out
searches never feed a value loss. (Unchanged; restated for one-page completeness.)

## 7. Tool inventory + status (2026-07-21)

| Tool | Status |
|---|---|
| `obs-tool export-value` | filter default-ON; `--no-filter` warns; tested |
| `fast_fens` | explicit `filter|nofilter` required; keys-mode for full-coverage joins |
| `build_canon_tight.sh` | the canonical shard→corpus builder (self-checking) |
| `build_hardneg_corpus.sh` | filtered + WDL-joined (hardneg-tight-v2) |
| `tools/cloud/build_corpus_from_shards.sh` | filtered (explicit flag) |
| `build_canon_corpus.sh` | DEPRECATED, exits unless FORCE_LEGACY=1 |
| `build_fulldist_corpus.sh` | parked; scale awk fixed 1.94→1.42; header warns |
| `relabel_depth.py` | curate = training-grade (ply gate added); non-curate warns pilot-only |
| `canon_filter` (rust bin) | superseded for shard corpora; header documents partial coverage |
| trainer bulletformat path | prints "must be pre-filtered" banner |
| `net_scale_check.sh` | hard pre-game gate in gate_canon.sh + gate_1536.sh |

## 8. When adding a NEW data pipeline

1. Decide the filter site (the layer holding raw scores + best moves) — add it
   there, on by default.
2. State the label units and derive MULT from §2 — then verify with
   `net_scale_check.sh` on the first trained net; the check is the authority.
3. Real WDL or WDL=0-with-matching-MULT; never placeholders into WDL>0.
4. Add the §5 self-checks + a ledger row.
5. Update the §7 table and, if the contract itself changes, D46 first
   (decisions change in decisions.md; docs follow; code follows docs).
