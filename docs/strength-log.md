# Strength log

Audit trail behind every strength claim (design/14 §5, D25):
SPRT verdicts, gauntlet ratings, and tier-1.5 tactical-suite
scores, one row per run, appended by tools/sprt.sh,
tools/gauntlet.sh, and tools/tactics.sh --log. Never edit or
delete history; corrections get a new row.

| date | commit | kind | detail | result |
|---|---|---|---|---|
| 2026-07-10 | d206ea3 | tactics | tactics-smoke.epd, movetime 1000ms, Hash=16 | 28/28 (100%) |
| 2026-07-10 | d0253a4 | tactics | wac-revised.epd, movetime 100ms, Hash=16 | 145/200 (72%) |
| 2026-07-10 | d0253a4 | gauntlet | vs sf-elo1900 (SF16 UCI_LimitStrength), 100 games, tc=2+0.02, Hash=16, openings-8ply.epd (engines/runs/gauntlet-20260710-194657) | 64.0% = +99.95 +/- 73.07 Elo (nElo +100.17 +/- 68.10); ordo gap +102.0 +/- 39.3 |
| 2026-07-10 | 3e479be | tactics | wac-revised.epd, movetime 100ms, Hash=16 | 154/200 (77%) |
| 2026-07-10 | 3e479be | gauntlet | M4 rung (P6 exit audit): vs sf-elo{1700,1900,2000,2200} (SF16 UCI_LimitStrength), 50 games each = 200, tc=2+0.02, Hash=16, openings-8ply.epd (engines/runs/gauntlet-p6audit) | 62.7% overall (+119 -68 =13); vs sf-elo1900: 69.0% (+33 -14 =3); ordo (candidate anchored 2000): sf-elo1700 1775.7 +/- 92.6, sf-elo1900 1858.1 +/- 63.1, sf-elo2000 1890.2 +/- 83.3, sf-elo2200 2079.4 +/- 64.1 — candidate between the 2000 and 2200 rungs; M4 >= 1900 passed (UCI_Elo-anchor caveat per gauntlet.sh) |
| 2026-07-11 | d2fd281 | selfplay | P7 exit audit (doc 12 step 7): 100 games self-play @ 10+0.1 sudden death, Hash=16, openings-8ply.epd, concurrency 7, RUSTFLAGS=x86-64-v3 (engines/runs/p7audit-forfeit-20260711-105348) | ZERO time forfeits; 100/100 real terminations (66 mates, 15 adjudication, 13 threefold, 4 insufficient, 2 fifty-move); max move time 2.16s; Elo +6.95 +/- 48.44 (self-play, expected ~0) |
| 2026-07-11 | d2fd281 | match | P7 exit audit (D25 non-regression): PVS+aspiration+full-qsearch+hygiene+clock-TM vs P6 baseline (f120450), 220 games @ 8+0.08, Hash=16, openings-8ply.epd, both RUSTFLAGS=x86-64-v3 (engines/runs/p7audit-vsp6-20260711-110253) | +65.51 +/- 42.00 Elo (nElo +73.72 +/- 45.91), LOS 99.92%, W-L-D 96-55-69 = 59.32%, Ptnml [10,19,24,34,23]; NON-REGRESSION PASS — CI lower bound +23.5 > 0, a significant gain; bench node-reduction at equal depth 334.7M -> 10.1M @ d6 (33x fewer) |
