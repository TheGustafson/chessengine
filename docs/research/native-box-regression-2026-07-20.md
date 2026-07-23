# Why the SF16 equal-time gap widened −41 → −130 on the faster box (2026-07-20)

**Question (user meta-goal):** on WSL2 the ARCHON-vs-SF16 equal-time (10+0.1)
gap was −41.5; on the new native-Arch box (7700X, AVX-512) it is ~−130. Why did
a *faster* box make us look *worse*?

**Answer (confirmed): it didn't make ARCHON worse — it made both engines search
~3 ply DEEPER at equal time, and depth is exactly where ARCHON's small per-node
eval deficit compounds.** The −130 is the honest deep-search gap; the −41 was a
shallower-search artifact of the slower box.

## Evidence

### 1. Equal depth at equal time (PGN depth analysis)
At 10+0.1 on native, ARCHON and SF16 reach the **same** median depth 19 (SF16
+0.3 ply mean), same median time/move. On WSL2 both reached ~depth 16. So the
box bought depth (16→19), not a speed advantage for either side. A gap at equal
depth ⇒ eval, not speed.

### 2. Fixed-depth sweep — the gap is a function of DEPTH (the core test)
ARCHON (d40-uho-720sb) vs SF16, fixed depth, 80–100g each, UHO book:

| fixed depth | ARCHON Elo vs SF16 |
|---:|:---|
| 6  | **+62** |
| 8  | **+53** |
| 10 | −5 |
| 12 | **+22** |
| 14 | **+19** |
| 16 | **−31** |
| 18 | *(pending)* |
| ~19 (equal-time) | **−129** |

ARCHON is **even-to-ahead through depth 14**, crosses to −31 by 16, and −129 by
19. **The depth-16 fixed point (−31) reproduces the WSL2 equal-time result
(−41) within noise** — same depth, same gap, different box. That is the smoking
gun: the box changed the *operating depth*, not the underlying strength.
(±68–76 Elo/point at these game counts, so individual wobbles like the depth-10
dip are noise; the shallow-vs-deep contrast is ~150–190 Elo, far above it.)

### 3. Loss mechanism — GRADUAL, not tactical (points to eval, not search)
Of 178 ARCHON losses in the native −129 games: **154 (86.5%) are gradual
positional declines**, only 24 (13.5%) sharp (≥2.0cp single-move eval crash from
a non-losing position). Median max single-move eval drop 1.39. ARCHON gets
slowly out-positioned, it does not blunder tactically. Matches the earlier
sf16-gap-diagnosis (0.2% blindspots, gradual per-node disadvantage).

## Interpretation

Coherent single story: **ARCHON's D40 eval is rich per-node** (enough to beat
SF16 at shallow search, where static judgment dominates) **but carries a small
systematic per-node deficit vs SF16** (distilled-from-SF16, slightly lossy).
Over deeper trees that per-node deficit compounds into gradual positional loss.
The fast box searches deeper at equal time, giving the deficit room to express —
hence −41 (depth 16) → −130 (depth 19).

**Strategic implication (validates the primary plan):** a *better eval* raises
the whole curve and shrinks the deep deficit → the SF18-canonical retrain is the
right lever. Search tuning is a minor lever here (only 13.5% tactical losses).

## Honest caveat

Fixed-depth does not *perfectly* separate eval from search: "depth 18" is a
different tree for each engine (different pruning), so some of the deep deficit
could be ARCHON's search producing a lower-quality depth-18 tree rather than the
eval alone. The gradual-loss signature argues against a large search component,
but the fully rigorous separation is the firewalled test (ARCHON-search +
controlled eval vs a top engine, fixed nodes — `what-holds-archon-back` §
"decisive confirming experiment"), still un-run. Confidence: **high that the
box-change is a depth effect; strong (not proven) that the deep deficit is
eval-dominated.**
