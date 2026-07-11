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
