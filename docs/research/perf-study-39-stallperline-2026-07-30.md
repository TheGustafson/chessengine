# Perf study 39 — the stall-per-line disagreement, closed

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, 32 KB L1d,
1 MB L2/core, 32 MB L3 · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Base:** `main` at `ad6d459` ·
**Artifacts and instruments:** `docs/research/perf-artifacts-39/`,
`tools/perf39/`

**Commission:** studies 29 and 31 priced the same quantity — the cost of one
64-byte line of accumulator streaming — in opposite directions on the same
night, agreed on misses per line to 8%, and disagreed **5.2×** on stall cycles
per line (29: **0.45**, subtractive; 31: **2.36**, additive). Both deferred it;
both said "close this first". It is the whole width of accumulator fusion's
bracket, +48 cycles/node (a loss) to −136 (+4.7 Elo). Close it and decide
fusion.

---

## ★ HEADLINE

*(filled below)*

---
