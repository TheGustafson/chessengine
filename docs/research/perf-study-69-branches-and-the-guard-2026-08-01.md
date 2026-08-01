# Perf study 69 — the branch-mispredict profile the engine never had, and the third `qtrace` form swept exhaustively

**Date:** 2026-08-01 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB L1i,
32 KB L1d, 1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`e9501029`**
(study 68 merged + its handoff) · **Net/arch:** D48X @ `network_scale=600`,
`sota,arch-d48x`, ACC 1280 · **Instruments:** `tools/perf69/` (nine new;
`tools/perf67/` and `tools/perf68/` reused, not rebuilt).

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes** on this base (study 68's frame was
5,001,560 — the tunable revert moved the tree), one thread, 16 MB TT, pinned to
core 2. Every binary is PGO (`tools/build_pgo.sh sota,arch-d48x <name>
engines/canonical/net.cenn`, eval-file mandatory). A GPU trainer and a
continuous match shared the box throughout; **no `cycles:u` is quoted as a
verdict** and every speed claim is a within-round paired median with its sign
count against a **byte-identical binary null measured in the same panel**.

---

## ★★★ 1. THE PRE-REGISTRATION (written and committed before the first panel)

### 1.1 The arm, and its price, measured before it was built

Study 68 §6 handed over `attacks::TABLES` as *"60 references inside
`search::<NonPv>` alone, 22 of them the `Once` state word"* and named the fix.
**A reference count is not a cost** — that is study 68 §4.3's own lesson, where a
sequence counted at ~8 instructions measured −0.75. So the guard was priced
first, by two independent instruments:

| instrument | what it answers | result |
|---|---|---|
| `tools/perf69/refoffs69.sh` | which BYTE of the static each reference touches | 22 refs to `TABLES+0x1040` (the `Once` state), 22 to `+0x0` and 16 to `+0x1000` (magic data) |
| `tools/perf69/guardcost69.sh` | how many times the guard EXECUTES per node, whole binary | **4.98 dispatched ops/node** across 15 symbols |
| `tools/perf69/guardsweep69.sh` | every guard-shaped static read in the binary | **4.21 ops/node total**, of which TABLES **3.40**, `THREAT_TABLES` 0.65, `CUCKOO` 0.15 |

Half the 22 sites in `search::<NonPv>` never execute at all (they are in cold
blocks); the two hottest execute 0.78 and 0.98 times per node.

**The sequence, quoted from `objdump` on the shipped PGO binary, is four
instructions and not three:**

```
mov  <TABLES+0x1040>(%rip),%eax      <- Once state, acquire
test %eax,%eax
jne  <init>                          <- never taken
lea  <TABLES>(%rip),%rdi             <- and this one is the surprise
```

The `lea` is structural, not incidental: **x86 RIP-relative addressing has no
index register**, so `magics[sq]` cannot fold the static's address into its own
operand the way a register base can. A `&'static` held in a register (or a
struct field) costs one `mov` and then indexes for free.

**Pre-registered price: −3 instructions per executed site.** At 3.40–4.98
sites/node that is **−10.2 to −14.9 instructions/node = −0.19 % to −0.28 %** of
the frame's 5,236.1 i/node.

**Pre-registered verdict counter: `instructions:u`, and only it.** It is exact
at fixed nodes, its byte-identical null has held 0.000 % across five panels of
studies 67–68, and this arm's effect is entirely an instruction-count effect.

**Pre-registered falsifications:**

- a reading of **0.000 %** falsifies the execution-rate measurement (the sites
  would not be running);
- a reading **worse than −0.10 %** means the field read is not being kept in a
  register across the uses the guard used to cover, i.e. the `lea` was
  double-counted;
- **any node-count or eval-hash movement at all is a bug**, not a result: the
  change is a representation change with no arithmetic in it.

### 1.2 The branch-mispredict target, and what it is allowed to conclude

The commission prices mispredicts at ~15 cycles. **This engine has measured its
own exchange rate and it is 9.13** (study 40 §3, from `ex_no_retire.empty`
against `ex_ret_brn_misp` over 159 paired rounds). Every cycle figure below uses
9.13, and the arithmetic is stated so it can be re-derived rather than believed.

Deliverable: a mispredict profile **down to the branch site**, which nothing in
this series could produce. §3.

---

*(Sections 2 onward are written after the panels.)*
