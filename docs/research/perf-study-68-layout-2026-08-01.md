# Perf study 68 — the frontend wall, attacked as LAYOUT

**Date:** 2026-08-01 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, **32 KB
L1i**, 32 KB L1d, 1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at
**`e9c99aca`** (study 67 merged) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Instruments:** `tools/perf68/` (+ study 67's
`tools/perf67/`, reused).

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,560 nodes**, one thread, 16 MB TT, pinned to
core 2. Every binary is PGO (`tools/build_pgo.sh sota,arch-d48x <name>
engines/canonical/net.cenn`). No `cycles:u` appears as a verdict anywhere.

---

## 0. PRE-REGISTRATION (written and committed BEFORE the first panel)

Study 67 left a budget and a rule:

> **445.7 cycles of empty op queue ÷ 128.0 L1i line fills = at most 3.48 cycles
> (~4.7 instructions) per fill.** Price the arm before building it, and never
> price an instruction-cache arm in bytes.

### Arm 1 — the machine function splitter (`-C llvm-args=-split-machine-functions`)

**Pre-registered price: ZERO instructions per fill removed.** MFS is a
MachineFunctionPass that moves basic blocks LLVM's own PGO counters call cold
into a separate `.text.split.<fn>` section. It creates no call, no argument
setup, no spill: a hot→cold edge becomes a jump to a far address instead of a
fallthrough, and by construction those edges are the ones the profile says are
rarely taken. Anything above ±0.15 % on `instructions:u` (whose null is
±0.000 %) is therefore an unpredicted cost and must be reported as one.

**Pre-registered prediction on the deciding counter:** `ic_cache_fill_l2` DOWN.
This is the falsifiable part. Study 67 §4.3 concluded that *"the bytes it
removes were never FETCHED — a line with zero executed ops never enters L1i"*.
**If that is the whole story, MFS moves the fill counter by ~0** — it removes
the same kind of bytes. A drop can only come from the population that
explanation does not cover: **64-byte lines that carry hot AND cold bytes
together**, which are fetched for the hot part and waste the rest. So the sign
of this arm decides between two mechanisms, and either answer is a result.

**Pre-registered secondary:** `de_src_op_disp.decoder` DOWN. Study 67 measured
40.5 % of ops arriving from the decoder rather than the op cache and did not
attack it; op-cache utilisation is driven by density inside 32-byte windows, so
a compaction arm should move it. This counter is new in this study
(`tools/perf68/panel68.sh`) and has no prior null — its null is measured in the
same panel.

**Pre-registered gates:** node total 5,001,560 exact and eval hash
`2e1ede95…79c4` exact, on every binary. A layout change that moves either is a
bug, not a win.

### The null

`s68-b0n` is an independent full rerun of `tools/build_pgo.sh` on the same
source with `PGO_EXTRA_RUSTFLAGS` unset. **`cmp` clean against `s68-b0` —
byte-identical**, so (a) study 67's determinism result still holds, (b) the
`PGO_EXTRA_RUSTFLAGS` hook added to `build_pgo.sh` is inert when unset, and (c)
the panel's null is a second name for the same bytes, which is the only null
that cannot smuggle in a codegen difference.

---

*(Results follow. Everything above this line was committed before the first
panel ran.)*
