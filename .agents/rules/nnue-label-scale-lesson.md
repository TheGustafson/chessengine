---
name: nnue-label-scale-lesson
description: "NNUE training labels must be trainer-matched (cp = scale·atanh(q)), NOT an engine's cp-display formula — the Lc0 conversion bug that caused a wrong \"Lc0 unusable\" conclusion"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c0a1b520-8665-4b2f-85c5-35473d97415c
---

When converting a win-probability/Q value into a centipawn training label for a
sigmoid-loss NNUE trainer (bullet, scale=400), the label MUST be the **inverse of
the trainer's own sigmoid**: `cp = scale · atanh(q)` (so `sigmoid(cp/scale) =
(q+1)/2 = the true win prob`). Do NOT use an engine's cp-**display** formula.

**The bug (2026-07-12):** our Lc0 converter used Lc0's UCI cp-display formula
`cp = 90·tan(1.5637·best_q)` (calibrated to Lc0's OWN win-prob model) and then fed
it into bullet's `sigmoid(cp/400)` — a **mismatched double-sigmoid** that compressed
mid-range labels ~2.5× (q=0.5: ours 89cp vs correct 220cp) and squashed the effective
win-prob target toward 0.5. Symptom: **median |cp| ≈ 16** (suspiciously compressed),
low-contrast net that couldn't separate a queen-up from a rook-up position. This
produced net-1 (−405 vs net-0) and net-2, and I **wrongly concluded "Lc0 data is
fundamentally unusable via transforms."** The user corrected it: Stockfish trains
NNUEs on Lc0 `best_q` data at scale (SF18: 100B+ positions) — the failure was OUR
converter, not the data.

**Why:** a monotonic label transform (net-2's γ-expansion) can't fix a wrong SCALE —
it's still the wrong mapping. The fix is the correct mapping, tested as net-3
(`cp = 400·atanh(best_q)`; ledger corpus `lc0-test91-atanh400`).

**How to apply:** (1) for any Q→cp datagen, use `scale·atanh(q)`, not a display
formula; verify the label distribution isn't suspiciously compressed (median |cp| in
the tens-to-low-hundreds, not ~16). (2) Field check first (we correctly used `best_q`,
the search-best-move eval SF's converter defaults to — not root_q/played_q/result_q).
(3) SF's fuller pipeline also deblunders + Syzygy-rescores + reduces early-ply weight
+ schedules lambda (1→0.5) — secondary quality levers if the mapping fix alone is
partial. See [[implementation-runbook]].

**net-3 result (CONFOUNDED, 2026-07-12):** the atanh remap de-compressed the
mid-range (median |cp| 16→45) and RESTORED material ordering (knight<rook<queen),
confirming the mapping matters — BUT net-3 was still ~58 Elo weaker than net-1,
because we remapped by INVERTING our already-CLAMPED display cp: all q>0.982 had
been flattened upstream to cp=2500, so the inverted eval is capped at ~935cp
(queen only 415) → can't express big advantages. So the atanh hypothesis is NOT
cleanly tested; a **precise re-decode from raw best_q** (full-range, no clamp→invert)
is the clean test. Standing evidence: net-1/net-2/net-3 all lose to net-0 (SF), but
none is a clean full-range-atanh test. LESSON REINFORCED: never invert an
already-clamped/rounded label to change its scale — go back to the raw source.
