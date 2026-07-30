# Perf study 41 — the front end, measured

**Date:** 2026-07-30 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 16 GB ·
**Base:** `main` at `56fc4c3` (study 40 merged) · **Net/arch:** D48X @
`network_scale=600`, `sota,arch-d48x`, ACC 1280 ·
**Instruments:** `tools/perf41/` · **Artifacts:** `docs/research/perf-artifacts-41/`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, shipped
release profile (fat LTO, CGU = 1), `-C target-cpu=native`, non-PGO, pinned to
core 2 with `taskset`. Every binary in this study is node-identical at
5,001,562 and eval-hash-identical at `2e1ede9521624dd8` — gated, not assumed
(`tools/perf41/identalign41.sh`).

**The reference reproduces study 40 on both of its headline levels**, which is
the frame check that matters most here:

| | study 40 (n = 159) | this study (n = 110) |
|---|---:|---:|
| `ex_no_retire.empty:u` / node | 192.907 | *see §1* |
| `ex_ret_brn_misp:u` / node | 24.092 | *see §1* |

---

*(results sections written as measurements land)*
