# Perf study 63 — the three dropped rungs, settled on wall clock

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1d,
1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`16098195`**
(post-study-62 handoff) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Artifacts:** `docs/research/perf-artifacts-63/`

---

## ★ PRE-REGISTRATION — written and committed BEFORE any wall-clock number

Study 62 dropped three rungs on a **proxy** — `>= 2.0 i/node per KB of .text` —
and refused to relax it after its own front-end panel argued the proxy was too
strict. That refusal was right. This study does not re-argue the threshold; it
measures the quantity the threshold stood in for.

**The pair.** `p63-base` (today's `main`, shipped configuration) against
`p63-all` (`shape62e` + `dual62b`, i.e. all three dropped rungs together), both
**native + PGO with the canonical net**, which is the configuration the engine
ships in and the one the studies 61+62 handoff measured. The three rungs are
taken together because the combined contrast is the largest one available and
the individual rungs (−0.13 % of instructions each) are a factor of two below
what any feasible panel can resolve.

**The instrument.** `tools/wallclock_panel.py`, its pre-registered median-only
licence unchanged, byte-identical null interleaved inside every round.

**The round count, and why it is not 40.** The tool's 40-round default is
calibrated in its own docstring for the ~1 % contrasts of studies 52–58. Power
in a sign test scales as the SQUARE of the contrast, and this arm is
**−0.6513 % of PGO instructions**, which at the handoff's measured 2.08×
instructions-overstate-time ratio predicts **≈ −0.313 % of wall clock** — a
third the size, so ~9× the rounds. Pre-registered: **400 rounds** for the full
frame and **400** for `--startup-frame`. At the sd of ~1.0 % that studies 62's
panels measured, 400 rounds put the arm at ≈ 5 SE of the median and the sign
test at z ≈ 4.9.

**The decision rule.**

1. If the panel is **NOT LICENSED**, nothing is quoted and **nothing lands** —
   that is the honest "wall clock cannot resolve this" outcome and it is a
   result, not a failure.
2. If it is LICENSED, the startup frame is subtracted **in absolute seconds**
   (the two panel percentages are relative to different baselines and cannot be
   subtracted directly), and the **search-only** residual is what is quoted and
   converted to Elo at 120/doubling.
3. **Land the three rungs iff the licensed, startup-corrected search-time change
   is negative.** A positive or zero residual kills them, and study 62's cut
   stands on a better instrument than the one that made it.

**Box discipline.** `uptime` and the top-CPU census immediately before and after
every panel, printed in the artifact. If the box stops being quiet the panel is
void and is not salvaged.

**Guard counters.** `instructions:u` (null 0.000, load-robust) and
`ic_tag_hit_miss.instruction_cache_miss`, both with their nulls in the same
rotation, on the same PGO binaries.

*(Committed before the first wall-clock round was run; everything below this
line was written afterwards.)*
