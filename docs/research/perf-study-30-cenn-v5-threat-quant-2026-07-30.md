# Perf study 30 — CENN-v5: `threat_quant` is a property of the NET

**2026-07-30.** A correctness fix with a measured motivation, not an
optimisation sweep. One landed change, one measured cost, one rejected
implementation with its number, and three latent bugs found on the way.

---

## 1. The defect

D49 is a 400-superbatch run using quantisation-aware training with
`qat_threat_scale = 256` (`tools/nnue/runs/d49_1280.json`). QAT clips the
FullThreats table to ±127/256 = ±0.496 so it fits an i8 **at scale 256**. Every
deployment path was hardcoded to 64.

Measured on the real packed containers (`sha`-verifiable, script in §8):

| net | container | `max\|w\|` | i8 levels used | threat RMS |
|---|---|---|---|---|
| `d48y2-1280-sb190-ema` (no QAT) | CEN2 @ 64 | **127**/127 | **247**/256 | 4.473 |
| `d48x-1280-sb520-ema` (canonical, no QAT) | CEN2 @ 64 | 126/127 | 235/256 | 3.781 |
| `d49-sb100-native` (QAT @ 256) | CEN2 @ 64 | **32**/127 | **65**/256 | 5.639 |

25% of the i8 range; two bits thrown away on the channel that is 72.9% of our
input space and carries 86% of all accumulator quantisation noise. D49-sb100
measured **−125.1 ± 20.9 Elo** against D48X over a 336-game fixed-node screen,
while its float heldout R² (0.8757) sat **within 0.008** of D48X's.

**The damage is invisible to every metric except games**, because R² is computed
on the float net and we play the quantised one. That is the general lesson of
this study, and it is worth more than the fix: a float-side metric cannot see a
quantisation defect, and there was no gate anywhere in the pipeline that looked
at the integer weights.

### Why not just change the constant

`THREAT_QUANT = 256` globally would wreck every existing net. A net trained
WITHOUT QAT has threat weights spanning ±1.98, and 64 is simply the largest
scale that does not clip them (127/1.9778 = 64.2). At 255 those weights measure
**6.80% relative error against 5.58% at 64** — worse, not better
(`tools/nnue/requant_threats.py`'s own sweep). QAT nets want 256; non-QAT nets
want 64. **The scale is a property of the net.**

---

## 2. CENN-v5 — the layout, and why five

`CEN3` and `CEN4` were **already taken** — by `src/eval/nnue/simple_halfka.rs`
and `src/eval/nnue/protected_halfka.rs`, both routed by `src/uci.rs`'s magic
peek. Reusing `CEN3` for the SOTA container would have misrouted a SOTA net into
the HalfKA loader, which is precisely the collision a magic exists to prevent.
The commissioning brief said "CEN3"; the next free number is **`CEN5`**, and
that is what landed. (Deviation surfaced here rather than footnoted.)

The v2 header is **full**: all 64 bytes assigned, no reserved slot. That is why
one new field costs a magic. v5 therefore keeps bytes `0..64` byte-for-byte
identical in meaning and **appends**:

```text
CENN-v5 header (72 bytes) = the CENN-v2 header, with
0       4     magic = b"CEN5"
4       2     version: u16 LE (= 5)
...           (8..64 EXACTLY as CENN-v2: dims, ft/l1 quant, ft_shift,
               network_scale, payload_len, payload_checksum)
64      4     threat_quant: i32 — power of two, {64, 128, 256}
68      4     reserved: u32, MUST be 0 (non-zero is rejected)
```

The payload starts at 72. Everything after it — the PSQ1 and SKP1 trailers — is
unchanged.

**Appending rather than inserting is the safety property's implementation.** The
parser reads `0..64` through one shared path for both versions; only the header
LENGTH and the two appended words differ. There is no code path in which a v2
file is read differently than before.

A `CEN2` file has no field for it and therefore **means 64**
(`sota_head::THREAT_QUANT_CEN2`) — exactly what every engine assumed before the
bump.

The runtime consumes it as
`sota_head::threat_shift_for(q) = log2(256/q)` — the power-of-two frame ratio
the merge has carried since perf study 8 §4.6. `64 → 2` (the literal every
backend hardcoded), `128 → 1`, `256 → 0`. The 0.39% gain approximation is the
same size in both directions: 255/64 = 3.984 → 4, and 255/256 = 0.996 → 1, so
study 8's measurement (74.0% of positions identical, mean 0.14 SF cp, max 2.42,
r = 0.999999) covers the new value as well as the old.

`threat_quant_for(shift) = 256 >> shift` is the exact inverse over the accepted
domain, so the loader and the serializer cannot disagree about which frame a net
is in — the shift is the only stored form.

The serializer emits **the oldest container that can represent the net**: `CEN2`
when the frame is the one a v2 header implies, `CEN5` otherwise. Every net that
round-tripped to v2 bytes before this change still does, byte for byte.

---

## 3. THE SAFETY PROPERTY: loading a CEN2 net is bit-identical

All three pins reproduce **unchanged** on the canonical net
(`engines/canonical/net.cenn` → `d48x-1280-sb520-ema.cenn`, a CEN2 file), on
both backends.

| pin | value | how |
|---|---|---|
| `Bench:` (default features, depth 6) | **587683** | `CHECK_NNUE_GATE=1 bash tools/check.sh` |
| eval hash (wl50, 50 FENs) | **`5ca978e0…8b38`** | same run, **and** `tools/perf26/nnuepins26.sh` |
| d13 node total | **1922947** | same run, **and** `tools/perf26/nnuepins26.sh` |

- `CHECK_NNUE_GATE=1 bash tools/check.sh` → **All checks passed**, `Bench: 587683`.
- `tools/perf26/nnuepins26.sh` → *both NNUE pins reproduce (avx2 backend)*.
  This is the **x86-64-v3** build, i.e. the AVX2 kernels; a `target-cpu=native`
  test never runs them.
- `tools/perf30/probe30.sh` (`target-cpu=native`, **AVX-512**) reports
  `hash=5ca978e012da1bb0` and `nodes=5,001,562` on every arm below — the AVX-512
  backend is bit-identical too, and the node count is unchanged.

In-tree, `sota_loader::tests::cen2_implies_the_legacy_threat_frame` pins that a
serialized `CEN2` container is `HEADER_V2_LEN + payload` bytes long and loads
back with `threat_shift == 2`.

The Python half is proved by byte parity, not by argument: re-exporting
`d48y2-1280-sb190-ema-raw.bin` under the new `ArchSpec.threat_quant()`
derivation (QAT off ⇒ 64) produces a `quant.bin` whose sha256 prefix is
`a1d3f66388ab0f55b7aa6787b61d2488` — **identical to the shipped file**, same
138,320,792 bytes.

---

## 4. THE NEW PATH: full-range threat quantisation

Re-exported the live `d49-1280-sb100-ema-raw.bin` at the derived scale and
packed it as CEN5:

```
threat weights: n=77,721,600  RMS 0.08802  max|w| 0.4948      <- QAT's ±127/256 clip
  scale  64 : err 0.004098  (4.66%, 26.6 dB)
  scale 256 : err 0.001028  (1.17%, 38.7 dB)   -> 3.99x better
  clipped at scale 256: 0 weights (0.0000%)
```

| net | container | `max\|w\|` | levels | nonzero |
|---|---|---|---|---|
| `d48y2-sb190` (healthy reference) | CEN2 @ 64 | 127/127 | 247/256 | 67.5% |
| `d49-sb100` (the bug) | CEN2 @ 64 | 32/127 | 65/256 | 74.2% |
| **`d49-sb100-ema` repacked** | **CEN5 @ 256** | **127/127** | **255/256** | 80.6% |

Full range, 255 of 256 levels — better occupancy than the healthy reference, and
**3.99× less quantisation error** on 72.9% of the input space. The container is
138,283,648 bytes: exactly the v2 pack plus the 8 appended header bytes.

It loads and evaluates in both backends, and they agree:

```
avx2   (x86-64-v3) : e3ad62bd97c4f42d
avx512 (native)    : e3ad62bd97c4f42d
```

Artifact: `data/nnue/tmp/s30/d49-sb100-ema-q256.cenn` (+ `.cpol`). **Not a game
result.** D49's eval frame is 0.35× SF and needs calibration before any match
(`b2af17c`), so this is a quantisation-fidelity result only.

---

## 5. The measured cost of a net-dependent merge

**Frame** (quote it with every number): `tools/perf30/probe30.sh` — a verbatim
copy of `tools/perf26/probe26.sh` with only the log/bin/target paths renamed
(`diff` them). CGU=1 + fat LTO, `target-cpu=native`, `bench 16 1 100000
tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`, `PERF_CPU=6`.
Gate: `instructions:u` AND `branches:u` per node. **Cycles are not measurable on
this box** and are not quoted.

### 5.1 Null first

| arm | i/n | br/n | nodes | hash |
|---|---|---|---|---|
| `s30-base-a` | 5654.23 | 588.73 | 5,001,562 | `5ca978e012da1bb0` |
| `s30-base-null` (inert comment added to `sota_simd.rs`) | 5654.23 | 588.73 | same | same |
| `s30-base-b` (baseline again, ~40 min later) | 5654.23 | 588.73 | same | same |

**0.000% on both axes**, twice. The instrument resolves what follows.

### 5.2 The arms

| arm | i/n | Δ i/n | br/n | Δ br/n |
|---|---|---|---|---|
| baseline (`<< 2` immediate everywhere) | 5654.23 | — | 588.73 | — |
| **runtime vector shift** (`_mm512_sll_epi16(tv, count)`) | 5688.71 | **+34.48 (+0.610%)** | 588.73 | +0.00 |
| plumbing only, shift pinned back to the immediate | 5654.23 | **+0.00** | 588.73 | +0.00 |
| **const-generic specialisation (LANDED)** | 5657.00 | **+2.77 (+0.049%)** | 589.58 | **+0.85 (+0.144%)** |

Node count (5,001,562) and eval hash (`5ca978e012da1bb0`) identical on every arm.
The landed arm was measured three times from two independently written copies of
the same edit (`s30-cen5-constgen`, `s30-cen5-final`, `s30-cen5-restored` — the
third after a `git checkout` clobbered the unstaged working copy and it had to
be redone): 5657.00 / 589.58 all three times, to the digit.

### 5.3 The runtime shift is NOT free, and the mechanism is load folding

The obvious implementation — pass the shift as a loop-invariant vector shift
COUNT, since `vpsllw zmm,zmm,xmm` is the same single instruction as
`vpsllw zmm,zmm,imm8` — cost **+34.48 i/n**. The diagnostic arm isolates it
exactly: keeping the whole new plumbing (the extra parameter through
`activate_ft`/`activate`/the kernel table, the `SotaHead` field, the loader, the
`_mm_cvtsi32_si128` setup) but pinning the AVX-512 shift back to an immediate
returns **exactly** to the baseline 5654.23 i/n. **The plumbing costs zero. The
entire cost is one instruction form.**

`objdump` says why:

```
baseline / immediate     vpsllw $0x2,(%rcx,%r8,2),%zmm3     ← load FOLDED into the shift
runtime count            vpsllw %xmm0,%zmm4,%zmm4           ← register-only
```

`VPSLLW zmm, zmm, xmm/m128` takes its memory operand as the **count**, not the
data, so the register-count form cannot fold the accumulator load and LLVM must
emit a separate `vmovdqu64` per merge — 80 of them per eval at ACC = 1280 — and
the unroll collapses with it (the baseline shows ~40 distinct folded-load shift
sites; the runtime-count build shows 16 register-only ones).

This is a general fact worth carrying: **in this codebase, turning a SIMD shift
immediate into a runtime count silently costs a load, not just an operand.**

### 5.4 What landed, and what it costs

Const-generic specialisation — the named fallback, and the pattern study 25
landed for fold shapes at −79.3 i/n. Each backend's `activate_inner` is generic
over `SHIFT`, and the backend's `activate` wrapper branches once per eval:

```rust
match threat_shift {
    0 => activate_inner::<0>(out, pst, threat, stm),
    1 => activate_inner::<1>(out, pst, threat, stm),
    _ => activate_inner::<2>(out, pst, threat, stm),
}
```

Study 25's lesson was that this trick only pays at near-total coverage (it lost
elsewhere at 52%). Here coverage is **100% by construction**:
`threat_shift_for` rejects anything outside `{64, 128, 256}`, so the `_` arm is
`MAX_THREAT_SHIFT` and the match is exhaustive with no panic path. That
restriction is a deliberate performance contract, documented at
`threat_shift_for`: nothing coarser than 64 is ever wanted (it would quantise
worse than today) and nothing finer than 256 fits, so three arms is the whole
real domain — 64 legacy, 128 the measured non-QAT optimum, 256 the QAT scale.

**Residual cost: +2.77 i/n (+0.049%) and +0.85 br/n (+0.144%).** That is the
per-eval dispatch: one load of `self.threat_shift`, one compare, one
well-predicted branch, at ~0.85 evals per node. It is 3.5% of what study 25's
kernel work bought, and it buys correctness for a class of nets that currently
loses 125 Elo. Not free, and reported as such.

---

## 6. Rejected, with the number

| rejected | number |
|---|---|
| **Change `THREAT_QUANT` to 256 globally.** | Non-QAT weights span ±1.98: **6.80% relative error at 255 vs 5.58% at 64.** Worse for every shipped net. |
| **Runtime vector shift count** (`_mm512_sll_epi16`/`_mm256_sll_epi32`). | **+34.48 i/n, +0.610%.** Loses the folded load (§5.3). |
| **Reuse the `CEN3` magic**, as the brief specified. | Zero measurement needed: `CEN3` is `simple_halfka`, `CEN4` is `protected_halfka`, both live in `uci.rs`'s magic peek. Would misroute a SOTA net into the HalfKA loader. |
| **Rebase the stored threat weights into the FT_QUANT frame at load** (so the merge needs no shift at all). | Perf study 8 §4.6, unchanged: saturates 0.031% of them at i8, **max 205 SF cp**. |
| **Widen the shift to all 9 representable values (`threat_quant` 1..256).** | Not measured — refused on the study-25 coverage rule. Nine monomorphisations of the hottest loop for six values no training configuration produces. One-line to widen if that ever changes. |
| **Two arms `{0, 2}` instead of three** (dropping `threat_quant = 128`). | Not measured. Would save at most ~0.85 i/n (one compare) and would delete the measured non-QAT optimum (1.65× less error, `requant_threats.py`). Judged not worth it — stated as a judgement, not a measurement. |
| **Specialise at net load into a function pointer on `SotaHead`.** | Not measured, rejected on mechanism: on the statically-dispatched AVX-512 build `activate` inlines fully into `forward_into`; a stored fn pointer would de-inline the hottest kernel to save one predicted branch. |

---

## 7. Three latent bugs found on the way (all fixed)

1. **`tools/nnue/requant_threats.py` used `ArchSpec.full()`** — `l2 = 128`,
   `factorize = False`. Every gen-2 run has `l2 = 16` and `factorize = True`, so
   for all of them the script wrote a wrongly-shaped `quant.bin` **and skipped
   the factoriser merge**. It now takes `--run-spec <run.json>`.
2. **The run-spec → `ArchSpec` mapping existed only inside `pump_train.py`**, so
   any re-export tool had to re-derive it and could silently get a different
   arch. It is now `ArchSpec.from_run_spec()`, read by both.
3. **`spec.QUANT_THREAT` was free to disagree with `qat_threat_scale`** — the
   root cause. `ArchSpec.threat_quant()` now *derives* the export scale from
   `qat_threat_scale` whenever QAT is on, and raises if it is not an integral
   power of two in `1..=256`. With QAT off it returns the module default, which
   is why the D48Y2 re-export is byte-identical (§3).

### And a detector, because the float metrics could not see this

`netpack.py` now **refuses** to pack a threat table whose i8 peak does not reach
half the range:

```
netpack: …/d49-1280-sb100-ema-quant.bin: threat i8 peaks at 32/127 at
threat_quant=64 — at least one bit of the i8 is unused, which is what a QAT net
exported at the legacy scale looks like (perf study 30: -125.1 Elo, invisible to R2).
```

Verified against the **live, shipped** D49 `quant.bin`: it dies and writes
nothing. `D40_ALLOW_THREAT_UNDERRANGE=1` overrides. This is arithmetic, not a
judgement call — a table that cannot reach half the range was quantised at least
2× too coarsely — so it refuses rather than warns. It is the check that would
have caught this at hour 0.

---

## 8. Reproduce

```bash
# the safety property (both backends)
CHECK_NNUE_GATE=1 bash tools/check.sh        # Bench: 587683 + both NNUE pins
bash tools/perf26/nnuepins26.sh              # the same two pins, x86-64-v3 / AVX2

# the cost
PERF_CPU=6 bash tools/perf30/probe30.sh <label>     # log: docs/research/perf-artifacts-30/probes.txt

# the new path, end to end
python tools/nnue/requant_threats.py \
    data/nnue/experiments/d49-1280/d49-1280-sb100-ema-raw.bin \
    /tmp/q256.bin --scale 256 --run-spec tools/nnue/runs/d49_1280.json
D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 D40_THREAT_QUANT=256 \
    python tools/nnue/netpack.py /tmp/q256.bin out.cenn d40
```

The i8-occupancy figures in §1 and §4 come from a 40-line header+slab reader;
`docs/research/nnue-arch-artifacts/net_anatomy.py` is the tracked equivalent
(CEN2 only — it asserts on the magic, which fails loudly rather than misreading
a v5 file).

---

## 9. Files

| file | change |
|---|---|
| `src/eval/nnue/sota_loader.rs` | `CEN5` magic/version/length, the appended parse, `BadThreatQuant`, one shared `0..64` path, `serialize` (was `serialize_v2`) emits the oldest container that fits, 3 new tests |
| `src/eval/nnue/sota_head.rs` | `THREAT_QUANT_CEN2`, `threat_shift_for`/`threat_quant_for`/`MAX_THREAT_SHIFT`, `SotaHead::threat_shift`, `activate_ft`/`policy_trunk`/`merge_transformers` take the shift |
| `src/eval/nnue/sota_simd.rs` | all three backends' `activate` take the net's shift; avx2/avx512 kernels are const-generic over it; dispatch parity tests now sweep shifts 0/1/2 |
| `src/eval/nnue/mod.rs`, `src/uci.rs` | magic peek accepts `CEN5` |
| `src/policy.rs` | passes the net's shift to `policy_trunk` |
| `tools/nnue/netpack.py` | `D40_THREAT_QUANT` env-overridable, emits v5 when ≠ 64, the under-range refusal, `strip_d40_threats` v5-aware |
| `tools/nnue/rescale_net.py` | v5-aware header length |
| `tools/nnue/requant_threats.py` | `--run-spec`, correct arch + shapes, no more monkeypatch |
| `jax/archon_jax/spec.py` | `ArchSpec.threat_quant()` (derived), `ArchSpec.from_run_spec()` |
| `jax/tools/pump_train.py` | uses `from_run_spec` |
| `design/11-evaluation.md` | §12.6 CENN-v5 |
| `tools/perf30/probe30.sh` | probe26 with renamed paths |

Pre-existing warnings NOT introduced here, verified against `b9ed7c6`: dead
`sota_loader::read_i8s`, dead `sota_simd::NNZ_IOTA`, and five clippy lints in
`aligned.rs` / `simple_halfka.rs` / `protected_halfka.rs`. None are in a file
this study touched; `tools/check.sh` clippies default features only, so the
`sota` build has never been lint-clean.
