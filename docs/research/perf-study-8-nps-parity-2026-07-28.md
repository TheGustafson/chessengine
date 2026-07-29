# ARCHON perf study #8 — closing the NPS gap to SF18: what parity actually is, and the first bit-exact instalment

Date: 2026-07-28. Eighth commissioned study in the chain (1:
`old/docs/research/threat-channel-perf-study.md`, 2–3: `perf-study-{2,3}-*.md`,
4–7: `perf-study-{4,5,6,7}-*.md`, plus `perf-impl-1-threats-and-icache.md`).
Raw artifacts and instruments: `perf-artifacts-8/`.

**Unlike 1–7 this study was licensed to change code, and it did.** Branch
`perf-study-8-nps-parity`. **Six changes are kept and committed** (§4.1–4.6,
**−8.02%** of all engine instructions together); **six more were implemented,
measured and reverted** (§7.1, §7.2, §7.7–§7.10); three were implemented purely
as pricing instruments and reverted (§4.3, §4.5, §6.1). Four of the six kept
changes are **eval bit-exact or node-identical**; two are deliberate
quantisation changes carrying a measured fidelity budget (§4.5, §4.6) and the
gate pins were moved with them in the same commits. The starting pins were
(wl50 hash
`d82caf8a…8163`, d13 `1,972,607`); they now stand at
**`5ca978e0…8b38` / `1,922,947`**, and
`CHECK_NNUE_GATE=1 bash tools/check.sh` exits 0 with `Bench: 587683`.

**Binaries in the worktree** (all PGO, `sota,arch-d48x`, canonical net):
`engines/bin/archon-s8-base` (HEAD before this study),
**`engines/bin/archon-s8-v3` (HEAD after, the deliverable)**, plus the per-arm
binaries named in §4 and `arms.txt`. `archon-s8-base-dbg` is the same PGO build
with line tables, used for `addr2line`/`srcline` work only.

**Configuration under study.** Canonical net `engines/canonical/net.cenn` ->
`d48x-1280-sb520-ema.cenn`, arch `sota,arch-d48x` (L1 1280 / L2 16 / L3 32,
8 buckets), built with `tools/build_pgo.sh sota,arch-d48x <out>
engines/canonical/net.cenn` — PGO, `target-cpu=native`, AVX-512 VBMI2 kernels.
Opponent: `engines/bin/stockfish18` (official x86-64-**vnni512** — the same
ISA class, so this is a native-vs-native comparison, not the AVX2-twin path
studies 2/3/7 used for callgrind schedules).

**Box state.** Quiet (load 0.2–0.4, 6 GB free) for the baseline counter rounds
and the flat profiles; **contended from ~15:55** (the user launched an
`archon` vs `stockfish18` match, load ~10, 2–4 GB free) for the later arms.
Consequence, stated once and applied everywhere: **every cycles and NPS figure
in this document is a band and carries no claim.** All conclusions ride on
`instructions:u`, which reproduced at **0.0000% spread** across six interleaved
rounds, on static `objdump`/`nm` counts, and on node totals.

---

## 0. HEADLINE

**(1) The parity target is 1.080×, and we are at 1.643×.** Decomposing SF18's
own instruction stream, **32.0% of it scales linearly with the big net's
accumulator width**. Carrying D48X's 1280 instead of SF18's 1024 therefore
justifies `1 + 0.25 × 0.320 = 1.080×` — an **8% premium, not the 25% that
1280/1024 suggests**. Parity for D48X is **≤ 4,677 instructions/node**. We
measure **7,116.6**, so the width-normalised residual is **1.522×** and closing
it means removing **34.3% of our instructions**. An independent derivation
using *our* width-dependent share instead of SF's lands at 1.517× — the two
routes agree to 0.3%.

**(2) The residual is 59% eval, 18% make/unmake, 14% shell, 6% memcpy.**
Instruction-attributed profiles of both engines, native AVX-512 on both sides:

| block | ARCHON i/n | SF18 i/n | ratio | share of the 2,438 i/n excess |
|---|---:|---:|---:|---:|
| **eval layer** | 3,596.7 | 1,811.5 | **1.985×** | **+1,439 (59%)** |
| make / unmake board | 782.8 | 346.0 | **2.262×** | +437 (18%) |
| search shell (search + qsearch + movepick + histories + see_ge + movegen) | 2,389.0 | 2,054.4 | **1.163×** | +335 (14%) |
| libc accumulator copies | 157.0 | ~0 | — | +157 (6%) |
| startup + unattributed | ~191 | ~119 | — | +70 (3%) |

**Our SEARCH SHELL is at 1.16× SF18 and is not the problem.** This corrects a
standing framing: study 7 read `next_move` and `search::<NonPv>` as "the icache
story" and ranked a code-size diet #3. Our movepick is *split* across
`init_quiets`/`init_captures`/`next_move` where SF18 has one `next_move`, so a
per-symbol comparison overstates us; totalled, movepick is 667 i/n against
SF18's 734.

**(3) Landed: −8.02% of ALL engine instructions, in six changes.** On the
matched 15.0 M-node wl50 panel (`counters-v3-wl50.txt`, three arms interleaved,
`instructions:u` spread 0.0000%):

| per node | before | after | SF18 | parity target |
|---|---:|---:|---:|---:|
| instructions | 7,116.6 | **6,546.0** (−8.02%) | 4,329.6 | 4,677 |
| ratio vs SF18 | 1.644× | **1.512×** | — | 1.080× |
| residual vs parity | 1.522× | **1.400×** | — | 1.000× |
| branches | 797.1 | **764.1** (−4.14%) | 394.7 | — |

**23.4% of the distance to the derived parity target, closed.**

| # | change | Δ instr | class |
|---|---|---:|---|
| 1 | `l1_dot`: the group's four FT bytes as one unaligned 32-bit load, deleting a bounds check LLVM could not hoist out of an 11-instruction loop run **42.79×/node** (§4.1) | **−1.77%** | bit-exact |
| 2 | `activate`: `\|4t\|` as `slli::<2>(abs(t))` (§4.2) | **−0.49%** | bit-exact |
| 3 | The Wave-G threat map filled **on demand**; ~52% of nodes never ask (§4.3) | **−1.01%** | node-identical |
| 4 | PSQ1 accumulate: ymm shape **and** a register home — neither alone works (§4.4) | **−0.43%** | bit-exact |
| 5 | **The FT frame ratio carried as 4, not 255/64** — six SIMD ops per merge become one shift (§4.5) | **−2.43%** | eval, 0.14 SF cp mean |
| 6 | **The output-skip lane as an integer `vpdpbusd` dot** instead of a dense f32 one (§4.6) | **−1.89%** | eval, 0.94 SF cp mean |

Speed, **as a band only** — the box carried a user match throughout, so the
paired per-round statistic is the only admissible form: NPS ratio vs SF18
**0.635 → 0.686**, ranges [0.528, 0.649] and [0.600, 0.736].

**(4) The branch ratio is an EVAL number, and it is now explained.** 2.02×
whole-engine splits into **eval 3.17×**, make/unmake 2.64×, shell **1.31×**.
Nobody had decomposed it. The `l1_dot` check alone was 5.25 percentage points
of the whole-engine figure.

**(5) The single largest *named* eval mechanism is the split threat
quantisation, worth ~2.5% of all engine instructions, and it is a net-format
item.** Our threat transformer is quantised at `THREAT_QUANT = 64` and the
HalfKA one at `FT_QUANT = 255`, so `activate` must compute
`round(threat · 255/64)` per lane before summing: **6 extra SIMD ops per merge,
24 of the 48 instructions in the loop body.** SF18 has the identical
two-accumulator shape (`HalfKAv2_hm` + `FullThreats`, both 1024-wide,
`nnue_feature_transformer.h:337-360`) and merges them with **one
`vec_add_16`**. Per 64 output bytes we issue 48 instructions to SF's ~19.
A deliberately eval-changing arm that replaces the rescale with a plain add
measured **−2.71% engine instructions**; static accounting says 2.47%.

**(6) SIX changes were implemented and did not pay; all are reverted and all
are in §7 with the number that killed them** (−0.045%, **+1.31%**, nothing over
9 paired rounds, **+0.19%**, **+2.2%**, **+1.04%**). Every one of the four was chosen by reading a
percentage off an annotated instruction or off a code shape; every one of the
three that *did* pay was chosen by counting — a loop body times its trip count,
or a three-minute deliberately-wrong stub that prices the candidate before
anyone designs the correct version (`scripts/probe8.sh`). On this box
`perf annotate` percent on stores and compares is **skid**: it valued the
insertion sort's bounds checks at 9.7% of `init_quiets` (real: 0.045% of the
engine) and the threat-PSQ1 store round-trip at 12.4% of `at_full` (real:
fixing it **cost +1.31%**). §4.4 is the procedure that replaces it.

---

## 1. THE PARITY TARGET — deriving it instead of assuming it

The mandate's question: how much of our instruction excess is *justified* by
D48X carrying a 1280-wide accumulator against SF18's 1024-wide big net?

Width scales the accumulator work only. It does not scale movegen, movepick,
the search shell, `do_move`, threat *event generation*, or the L2/L3/output
tail. So the justified premium is

```
J = 1 + (1280/1024 − 1) · f_w      where f_w = the fraction of SF18's
                                   instructions that scale linearly in its
                                   big accumulator width
```

`f_w` is measured, not assumed, from SF18's own instruction-attributed profile
(`perf record -e instructions:u -c 500000`, so a sample is worth exactly
500,000 retired instructions and per-symbol share **is** per-symbol instruction
share):

| SF18 symbol | % of SF18 instructions | scales with 1024? |
|---|---:|---|
| `Network<1024>::evaluate` | 28.36 | yes × 0.95 — the FT transform, the lazily-applied accumulator chain and the `AffineTransformSparseInput<1024,16>` all scale; `fc_1` (15→32) and `fc_2` (32→1) are ~512 fixed MACs |
| `update_accumulator_refresh_cache<1024>` | 2.28 | yes |
| `double_inc_update<1024>` (both instantiations) | 2.18 | yes |
| `update_threats_accumulator_full<1024>` | 0.33 | yes |
| `update_accumulator_incremental<*,1024>` | 0.30 | yes |
| **f_w** | **32.03%** | |
| `Network<128>::evaluate` + `refresh_cache<128>` + `double_inc_update<128>` | 2.51 | **no** — scales with 128, and the dual net is an *efficiency*, not a width justification |
| `update_piece_threats` + `swap_piece` | 7.02 | **no** — O(events) |
| everything else (shell, do_move, loader) | 58.44 | no |

⇒ **J = 1 + 0.25 × 0.3203 = 1.0801**, band **1.075–1.084** for the
`evaluate`-tail assumption swept over 0.90–1.00.

**Parity for D48X = 4,330.6 × 1.0801 = 4,677 instructions/node** (band
4,657–4,693).

### 1.1 Cross-check from our own side

The same classification applied to *our* profile gives our width-dependent
share at 1280 as **38.56%** (`activate` 4.94 + `find_nnz` 0.75 + `l1_dot` 6.17
+ `skip_dot` 2.50 + the `at_full` fold ~10.64 + `current_full` ~5.34 +
`i16_deltas` 3.45 + `finny_refresh_into` 1.86 + `refresh_threat` 0.91 + the
accumulator memcpys 2.00). Shrinking only that block to 1024 gives a
1024-equivalent ARCHON at `7116.6 × (0.6144 + 0.3856 × 0.8) = 6,568 i/n`, i.e.
**1.517× SF18**.

Against the 1.522× the SF-side derivation gives, the two agree to 0.3%. They
are not the same computation — one prices the width premium at SF's efficiency,
the other at ours — and the agreement says the two engines devote nearly the
same *share* of their work to width-scaling code. **The lenient and strict
definitions of parity coincide here; there is no room to argue the target down.**

### 1.2 What parity means per block

Spending SF18's block costs through `J` gives the budget each of our blocks
would have at parity:

| block | SF18 i/n | width-justified budget | ARCHON actual | over budget |
|---|---:|---:|---:|---:|
| eval | 1,811.5 | **2,158** | 3,596.7 | **+1,439** |
| shell | 2,054.4 | 2,054 | 2,389.0 | +335 |
| make/unmake | 346.0 | 346 | 782.8 | +437 |
| libc copies | ~0 | 0 | 157.0 | +157 |
| misc | ~119 | 119 | ~191 | +70 |
| **total** | **4,330.6** | **4,677** | **7,116.6** | **+2,438** |

Width-normalised, our eval layer is **1.70×** SF18's, our make/unmake **2.26×**,
our shell **1.16×**.

---

## 2. GROUND TRUTH — matched fixed-node counters

`bench 16 1 300000 tools/nnue/calib/wl50.fens nodes`, one run of each engine
per round, `taskset -c 4`, `nice -n 19`, 6 rounds, quiet box.
Raw: `perf-artifacts-8/counters-baseline-wl50.txt`; parse with
`scripts/parse8.py <log> sf18`.

| per node | ARCHON D48X PGO | SF18 | ratio |
|---|---:|---:|---:|
| **instructions (deterministic)** | **7,116.6** | **4,330.6** | **1.6433×** |
| cycles (band) | 4,271.7 | 2,853.0 | 1.497× |
| branches | 797.1 | 394.7 | 2.0195× |
| branch-misses | 22.2 | 26.5 | **0.8376×** (we win) |
| NPS (per-round paired median) | 1.253 M | 1.906 M | **0.6565** |

Node totals 15,001,846 vs 15,010,238 (0.06% apart). `instructions:u` spread
across the six rounds: **0.0000%** on both arms.

**SF18's 4,330.6 i/n reproduces study 7's 4,330.6 exactly**, and study 3's
4,331.4 and study 2's 4,332.4 — four independent sessions, four digits. The
instrument is sound.

### 2.1 What moved since study 7 (D47V) — the width bought a real bill

| | study 7, D47V PGO (768) | this study, D48X PGO (1280) |
|---|---:|---:|
| instructions/node | 6,482.8 | **7,116.6** (+9.8%) |
| ratio vs SF18 | 1.497× | **1.643×** |
| branches/node | 783.5 | 797.1 |
| cycles/node | 4,588 (contended box) | 4,272 (quiet box) |

D48X costs +9.8% instructions per node over D47V **even after `perf-impl-1`'s
−6.17% landed in between** — so the raw arch step was worth roughly +16%. That
is the price the D48X promotion knowingly paid, and it is exactly the
"1280 = −28 Elo of equal-time speed" the pre-launch NPS gate priced.

---

## 3. WHERE THE INSTRUCTIONS ARE

Full tables: `perf-artifacts-8/block-decomposition.txt`,
`flat-{ours-d48x,sf18}-instructions.txt`, `flat-*-branches.txt`.

### 3.1 The two flagships, side by side

| ARCHON | % | SF18 | % |
|---|---:|---|---:|
| `search::<NonPv>` | 17.88 | `Network<1024>::evaluate` | 28.36 |
| `SotaHead::forward_into` | 16.55 | `Worker::search<NonPv>` | 19.51 |
| `ThreatContext::at_full` | 16.37 | `MovePicker::next_move` | 16.95 |
| `MovePicker::init_quiets` | 7.53 | `Position::do_move` | 6.62 |
| `Board::make_move_with` | 7.18 | `update_piece_threats` (both) | 5.65 |
| `HalfkaContext::current_full` | 6.67 | `Position::see_ge` | 2.82 |
| `Worker::do_move` | 3.82 | `Worker::qsearch<NonPv>` | 2.34 |
| `avx512::i16_deltas` | 3.45 | `refresh_cache<1024>` | 2.28 |
| `record_square_touch` | 2.72 | `double_inc_update<1024>` | 2.18 |
| **libc `__memmove_avx512`** | **~2.00** | `Network<128>::evaluate` | 1.83 |

### 3.2 `forward_into` opened up

`perf annotate` + `scripts/annot8.py` split the symbol at its phase boundaries:

| phase | % of `forward_into` | % of engine |
|---|---:|---:|
| `activate` — the pairwise FT merge over 4 × 1280 i16 | 29.84 | 4.94 |
| `find_nnz` — `vptestmd`/`vpcompressw` | 4.53 | 0.75 |
| **`l1_dot`** — sparse int8 matmul | **37.30** | **6.17** |
| `l1_dequant` | 0.26 | 0.04 |
| `l2_tail` (16→32) + `l3` (32→1) | 13.03 | 2.16 |
| `skip_dot` — DENSE 1280-lane u8 × f32 | 15.09 | 2.50 |

Two derived facts fall straight out and both correct standing beliefs. Both
rest on an **exact** count, not an estimate: §4.1's change removes exactly one
conditional branch per active group, so the measured branch delta **is** the
group count.

**(a) The engine executes 42.79 active FT groups per node ⇒ group density is
~36%, not the "~85–95% are zero" the `find_nnz` doc comment claims.**
`(4,075,674,469 − 3,861,673,358) / 5,001,629 = 42.79` branches/node removed =
groups/node. The instruction delta over the same arm is 128.36/node =
**exactly 3.000 per group**, which is what the disassembly shows the change
removed (`cmp`, `ja`, and a now-dead `mov`) — the arithmetic closes to three
decimals, so the group count is not an inference. Dividing by
evals/node = 0.366 (from `activate`'s 351.6 i/n over a 960-instruction
per-eval kernel) gives **116.8 active groups of 320 = 36.5% density**.

**(b) That kills the sparse-`skip_dot` idea before it is written.** At 36.5%
group density a 16-element block is nonzero with probability
`1 − 0.635⁴ = 84%`, so gating `skip_dot` on the nnz list saves 16% of 2.50% =
0.4% at best, for a mask-plumbing change. Carried option 6's `skip_dot` half is
**dead**. (The right fix for `skip_dot` is SF's: fold the skip lane in as the
`FC_0_OUTPUTS + 1`-th row of the *sparse* int8 matmul, which costs ~1/16 of
`l1_dot` instead of 2.50% — a CENN format rev, see §6.)

### 3.3 The libc block study 7 recorded as absent

Study 7 §6 wrote "`memset`/`memcpy`: absent from every profile above 0.4% — B4
stays MOOT". That is **wrong at D48X**, and the reason it looked right is an
instrument artifact: perf cannot symbolise glibc's ifunc-resolved bodies, so
the block appears as bare `0x189e4d`-family addresses with `libc.so.6` as the
only clue. Resolved by `objdump`ing libc at those addresses and by
`perf report -s srcline` on a line-table PGO build:

* `memmove-vec-unaligned-erms.S` ≈ **2.00% of all engine instructions**
* `memset-vec-unaligned-erms.S` ≈ 0.10%
* `malloc` 0.12%

`scripts/libcalls8.py` walks the disassembly and names the callers. Every hot
one copies **2560 B = 2 perspectives × 1280 i16 = one accumulator pair**:
`Worker::do_move` ×2, `at_full`, `current_full`, `finny_refresh_into` ×2,
`ThreatContext::refresh` ×2, `apply_at` ×2 (the by-value `refresh_threat`
return). **B4 is no longer moot — but it is memMOVE, not memSET**: the dead
`let mut ft = [0u8; L1]` zero-fill in `forward_into` is real but is inside the
0.10% memset block and is not worth touching.

### 3.4 Branches, decomposed for the first time

| block | ARCHON b/n | SF18 b/n | ratio |
|---|---:|---:|---:|
| eval layer | 358.6 | 113.3 | **3.166×** |
| make / unmake | 91.7 | 34.7 | 2.643× |
| search shell | 286.0 | 217.6 | 1.314× |
| **whole engine** | **797.1** | **394.7** | **2.0195×** |

The mandate called branches/node "the most under-explained number in the
chain". It is an eval-layer number: our SIMD kernels are short-trip loops with
per-iteration index bookkeeping where SF's are register-blocked and unrolled.
It also stays a *count*, not a cost — branch-misses/node is **0.84× SF18's**,
and study 2 closed the branch-prediction front for good reasons. The one place
it converted to instructions was `l1_dot`'s bounds check, §4.1.

---

## 4. WHAT WAS IMPLEMENTED

Arm harness: `scripts/arm8.sh <name>` — PGO-builds the arm, checks the wl50
eval hash against `d82caf8a…8163`, checks the wl50/d13 node total against
`1,972,607`, then measures `instructions:u` on a fixed
`bench 16 1 100000 wl50 nodes` tree. Raw: `perf-artifacts-8/arms.txt`.

| arm | eval hash | d13 nodes | instructions | Δ vs previous kept | verdict |
|---|---|---|---:|---:|---|
| `base0` (HEAD) | OK | 1,972,607 | 36,633,073,946 | — | baseline |
| **`l1nobc`** | **OK** | **1,972,607** | **35,991,058,878** | **−1.752%** | **KEPT** |
| `sort` | OK | 1,972,607 | 35,974,886,358 | −0.045% | reverted (§7.1) |
| `psqtreg` | OK | 1,972,607 | 36,461,356,237 | **+1.306%** | reverted (§7.2) |
| `uniquant` | **CHANGED** | 1,961,283 | 35,014,708,233 | −2.71% (tree differs) | instrument only (§6.1) |

### 4.1 KEPT — `l1_dot`: one bounds check, −1.75% instructions / −5.25% branches

The AVX-512 sparse matmul's group loop, disassembled from the shipped PGO
binary:

```
movzwl 0xee(%rsp,%rsi,1),%ecx        ; nnz[i]
cmp    $0x13f,%ecx                   ; <-- bounds check on ft[group*4 + 3]
ja     <panic_bounds_check>          ; <--
mov    %ecx,%r8d
vpbroadcastd 0x378(%rsp,%rcx,4),%zmm1
shl    $0x6,%r8d
lea    0x2(%rdx,%rsi,1),%rcx
add    $0x2,%rsi
vpdpbusd (%rax,%r8,1),%zmm1,%zmm0
cmp    $0x2,%rcx
jne    <loop>
```

Eleven instructions, of which three do the actual work — after the change the
body is **eight** (`movzwl`, `vpbroadcastd`, `shl`, `vpdpbusd`, `lea`, `add`,
`cmp`, `jne`): the `cmp`/`ja` pair goes, and so does a `mov` that only existed
to keep a second copy of the index alive across the check. The check exists
because the kernel read the group's four bytes as
`i32::from_le_bytes([ft[g*4], ft[g*4+1], ft[g*4+2], ft[g*4+3]])`, and LLVM
cannot see that `find_nnz` — the only producer of `nnz` — emits indices in
`0..ACC/4`. It could not hoist the check out of the loop either, because the
bound depends on the loaded value.

The fix reads the four bytes as **one unaligned 32-bit load** through the
pointer already inside the kernel's `unsafe` block, with the range contract
stated in a `# Safety` clause and enforced by a `debug_assert` in the safe
wrapper — so the invariant is checked once per group across the entire test
suite instead of once per group in the shipped engine. The value is bit-identical
(`from_le_bytes` of four consecutive bytes *is* an unaligned little-endian
32-bit load on x86), and both the AVX-512 and AVX2 backends use it.

Measured on an identical 5,001,629-node tree:

| | base0 | l1nobc | Δ |
|---|---:|---:|---:|
| instructions | 36,633,073,946 | 35,991,058,878 | **−1.752%** |
| branches | 4,075,674,469 | 3,861,673,358 | **−5.251%** |

−214 M branches is **42.79 removed branches per node** and −642 M instructions
is **128.36 per node** — a ratio of **3.000**, exactly the three instructions
the disassembly shows the loop body lost. The arithmetic closes to three
decimals, which is what turns "42.79" from an inference into a measurement of
the engine's active-group count (§3.2a).

Verification: wl50 eval hash `d82caf8a…8163` (unchanged), 800-FEN eval hash
`53fc2de1…ad64` (unchanged), wl50/d13 total 1,972,607 (unchanged),
`cargo test --features sota,arch-d48x` 678 passed,
`CHECK_NNUE_GATE=1 bash tools/check.sh` clean with `Bench: 587683`.

### 4.2 KEPT — `activate`: `|4t|` in two instructions instead of three

The pairwise-FT merge has to rescale the threat accumulator by 255/64 before
summing it with the HalfKA one (§6.1), and built `4t` with two saturating adds
before taking the magnitude:

```
t4  = adds(adds(tv,tv), adds(tv,tv))      ; vpaddsw, vpaddsw
mag = mulhrs(abs(t4), 32640)              ; vpabsw, vpmulhrsw
```

`|4t| == 4·|t|` for every representable `t`, so taking the absolute value first
reaches `4t` with one shift:

```
mag = mulhrs(slli::<2>(abs(tv)), 32640)   ; vpabsw, vpsllw, vpmulhrsw
```

`vpsllw` wraps where `vpaddsw` saturates, which is exactly as safe here — both
forms already rest on the documented `|t| ≤ 8191` bound (`4·8191 = 32764 <
2^15`) and its debug assert. Four merges per loop body ⇒ −4 of the body's 48
instructions. **−0.49% engine instructions**, both eval hashes and the d13 node
total unchanged.

### 4.3 KEPT — the Wave-G threat map is now filled on demand

`Board::set_check_info` refilled the whole D30 threat map — every square
attacked by the side-not-to-move, split by attacker type — inside **every**
`make_move`: a pawn shift plus one magic lookup per enemy knight, bishop, rook
and queen, plus the king. Priced by stubbing it out (`scripts/probe8.sh`,
native non-PGO pair): **37,142,878,513 → 36,355,510,273 = 2.12% of all engine
instructions.**

Only two sites read it: `MovePicker::init_quiets` (the quiet scorer's threat
escape/entry and offense terms) and the N10 reverse-futility "is one of my own
pieces hanging" test. **No qsearch node touches either**, and neither does any
node that cuts before N10 — so a large share of the nodes that paid for the
fill never looked at it. Stockfish arrives at the same place from the other
end: it maintains its threat state incrementally in `do_move`
(`update_piece_threats`, 5.65% of its instructions) rather than refilling it.

The map therefore moves out of `StateInfo` into a lazy cache on `Board`, keyed
by the **raw Zobrist key**. Keying on the position hash rather than a
generation counter is what makes it cheap and safe: make, unmake and null-move
need no invalidation hook at all (a changed position is a changed key), and
returning to a previously-seen position reuses the fill. A key collision
(2^-64) can hand a stale ordering hint to a heuristic; it can never affect
legality, and the TT already rests on that bound. The cache uses `Cell` — safe
Rust, no `unsafe_code` exception — clones cold, and compares equal always,
since two boards are the same position whether or not either has materialised
its map. Dropping the two fields also shrinks `StateInfo` from 224 to 168
bytes, which shrinks the per-make copy.

**−1.01% engine instructions**; `make_move_with` 7.13% → 6.27% of the engine.
Node-identical, eval bit-exact, full D24 perft ritual and
`CHECK_NNUE_GATE=1 bash tools/check.sh` green.

The gap between the 2.12% priced and the 1.01% delivered is the read rate:
roughly 52% of nodes ask for the map. Making the two consumers cheaper — or
unifying this map with the NNUE FullThreats event stream, which is a second,
independent threat system (`record_square_touch`, 3.03%) — is where the rest
is (§5 item 8).

### 4.4 KEPT — the PSQ1 accumulate needed a ymm shape AND a register home

`psqt_add`/`psqt_sub` are ~3.0% of all engine instructions: they run once per
resolved feature in *both* the threat fold and the HalfKA fold. `BUCKETS` i32
is exactly 32 bytes — one `ymm` — and the generated code was **twelve**
instructions: an xmm group for lanes 0..3, a `vmovq` pair for 4..5, and two
scalar lanes.

Two independent defects, and **fixing either one alone loses**:

| | change | measured |
|---|---|---:|
| a | row as `&[i32; BUCKETS]` (via `try_into`) instead of a runtime-length slice | **+0.14%** instructions, +0.94% branches |
| b | accumulator hoisted into a local instead of a `&mut` out-param | **+1.31%** |
| **a+b** | **both** | **−0.43%** |

(a) alone fixes the vector shape but pays a length check on a three-instruction
body. (b) alone only adds copies, because the body it is trying to keep in
registers is still twelve instructions long. Together the body is one ymm
load-add-store and the accumulator stays in a register across the whole event
loop. Applied at all three sites: `resolve_into`'s threat delta, `psqt_fold`,
and the Finny cached refresh.

Also tried: reaching the row through a **raw pointer** to drop the length check
as well — the `l1_dot` trick from §4.1. That measured **+2.2%**; it costs LLVM
the aliasing facts the slice index gives it and de-optimises the surrounding
fold. The same instrument does not work in the same way twice.

### 4.5 KEPT — the FT frame ratio is 4, not 255/64 (−2.43%)

The two transformers are quantised differently — HalfKA at `FT_QUANT = 255`,
FullThreats at `THREAT_QUANT = 64` — so `activate` computed
`round(t · 255/64)` per lane before summing: abs, shift, mulhrs, compare-mask,
masked-sub, add. **Six SIMD ops, 24 of the merge loop's 48 instructions, ~2.2%
of ALL engine instructions.** SF18 has the identical two-1024-wide-accumulator
shape and merges with a single `vec_add_16`, because both of *its* transformers
share one frame.

The move that unlocks this is noticing the ratio **does not have to be exact**.
It is a gain on one eval channel. 255/64 = 3.984375 → 4 is a 0.39% change, well
inside the net's own quantisation noise, and it makes the merge `t << 2`.
One instruction. `THREAT_QUANT` is effectively 63.75.

This is an eval change, so it is *measured*. Against the exact form over the
800-FEN battery (`scripts/evaldiff.py`, new):

| | identical | mean \|Δ\| | p90 | p99 | max | Pearson r |
|---|---:|---:|---:|---:|---:|---:|
| **×4 (kept)** | **74.0%** | **0.14 SF cp** | 0.48 | 0.97 | **2.42** | 0.999999 |
| exact `4t − t/64` (`slli`/`srai`/`sub`) | 38.4% | 0.55 | 1.45 | 2.90 | 6.28 | 0.999992 |
| rebase the stored weights to the 255 frame | 34.6% | 3.97 | 11.59 | 30.92 | **204.83** | 0.998908 |

Two rejections worth keeping:

* **Rebasing the weights at load** is the obvious fix — it would make the merge
  a bare add. It fails on *range*: the trained net reaches |w| = 126 where only
  |w| ≤ 31 survives the rescale, so 24,334 of 77,721,600 weights (0.031%)
  saturate at i8 — and they are the **largest** weights, which is why the tail
  is 205 SF cp. Rejected on the no-blunders bar. The i8 range is a real
  constraint, and it is *why* the trainer picked a second frame in the first
  place.
* **The exact `4t − t/64`** costs an extra op and is **four times less
  faithful**, because `vpsraw` floors and that asymmetry is noisier than the
  0.39% of gain it corrects. Cheaper *and* more accurate is not the trade one
  expects; it is what the measurement said.

### 4.6 KEPT — the output-skip lane is an integer dot (−1.89%)

The D47V skip lane was a **dense 1280-element f32 dot** — `vpmovzxbd` /
`vcvtdq2ps` / `vfmadd` per 16 activations, ~240 instructions per eval — for the
one channel SF18 gets essentially free as the `FC_0_OUTPUTS + 1`-th row of the
sparse int8 matmul it already runs (its `L2Big` is **15**; the 16th output *is*
the skip lane).

We cannot fold it into `l1_dot` without a 17th output row, which means a
second, near-empty zmm accumulator and a doubled kernel. But we can pay in the
same currency: quantise the lane to i8 at load, one scale per bucket, and it
becomes a `vpdpbusd` dot — 64 u8 × i8 products per instruction, so 1280 lanes
cost 20 loads and 20 multiply-adds.

Integer accumulation also **deletes a constraint**. The f32 form had to pin a
16-partial-lane + halving-tree shape across every backend so they would agree
bit-for-bit; i32 addition is associative, so each backend now reduces however
is fastest and they still agree exactly. The parity test got stronger with it:
it asserts exact equality and pins `ft` at its 127 ceiling against both i8
rails, which turns the AVX2 backend's "`maddubs` cannot saturate"
(2·127·127 = 32258 < `i16::MAX`) from a comment into a tested claim.

Fidelity: **mean 0.94 SF cp, p90 1.93, p99 3.38, max 4.35, r = 0.999985** —
bounded and tail-free.

### 4.7 An observation about the three eval-changing wins

§4.5 and §4.6 are both *quantisation* changes, and both landed only because the
question asked was "how much accuracy does this cost, in cp, on 800 positions"
rather than "is this bit-exact". Bit-exactness is the right default gate — it
is what makes §4.1–4.4 cheap to trust — but it is a *gate*, not a *goal*, and
holding it as a goal is what kept ~4.3% of the engine's instructions parked in
this chain's earlier studies as "needs a retrain". Neither of these needed a
retrain. One is a constant; the other is thirty lines in the loader.

The three arms also each **reduced the d13 node total** (1,972,607 → 1,945,095
→ 1,922,947, −2.5% overall), i.e. the perturbed evals search slightly smaller
trees at fixed depth. That is not evidence they are *better* evals, and it is
not claimed as such — it is recorded because it means the per-node instruction
comparisons across those pins carry a tree-shape difference.

### 4.8 Why these paid when five similar-looking ones did not

The three kept changes were selected by **counting**: the `l1_dot` check from
the loop body (11 static instructions) and its trip count; the `activate` op
from the loop body (48 instructions, 4 merges); the threat map from a
**stub-and-diff price** (`scripts/probe8.sh`) taken *before* any design work.

The four that failed (§7.1, §7.2, §7.7, §7.8) were every one of them selected
by reading a percentage off an annotated instruction or off a code shape. Their
outcomes: −0.045%, **+1.31%**, nothing (4/9 rounds), **+0.19%**. This is study
3's lesson (`dynamic sample share on cheap ALU ops is not evidence`) recurring
in four new costumes, and it is now a *procedure*:

1. **Price first, design second.** A three-minute deliberately-wrong stub
   (`probe8.sh`) puts a number on a candidate before anyone writes the correct
   version. It cost 3 minutes and would have saved both §7.2 and §7.8.
2. `scripts/bcheck8.py` and `annot8.py` produce a **candidate list**, never a
   ranking. The ranking must come from `static instructions in the loop ×
   trip count`, or from a stub-and-diff.
3. On this box, `perf annotate` percent on **stores and compares is skid**.
   It valued the insertion sort's checks at 9.7% of `init_quiets` (real:
   0.045% of the engine) and the threat-PSQ1 store round trip at 12.4% of
   `at_full` (real: fixing it costs 1.31%).

---

## 5. WHAT PARITY WOULD COST, ITEM BY ITEM

Everything below is priced against the 2,438 i/n we are over budget.

| # | item | size | bit-exact? | status |
|---|---|---:|---|---|
| 1 | ~~Unify threat quantisation~~ — **DONE, §4.5** | **−2.43%** | eval change, 0.14 SF cp mean | **LANDED.** Needed no retrain and no format rev: the frame ratio is a constant. |
| 2 | ~~Fold the skip lane into `l1_dot`~~ — **DONE DIFFERENTLY, §4.6** | **−1.89%** | eval change, 0.94 SF cp mean | **LANDED.** Not folded into `l1_dot` (that needs a 17th output row and doubles the kernel) but paid in the same currency: i8 + `vpdpbusd`. Thirty lines in the loader. |
| 3 | `at_full`'s non-fold half — the per-eval threat replay plumbing | 5.73% of engine (408 i/n); SF18 has **no** equivalent stage | yes, in principle | hard; impl-1 already took the representation win |
| 4 | Accumulator copies through libc `memmove` — §3.3 | 2.00% (157 i/n) | yes | call sites named; not attempted |
| 5 | make/unmake at 2.26× SF18 (`make_move_with` + `Worker::do_move` = 783 i/n vs 346) | +437 i/n | yes | **never studied by anyone in this chain**; §4.3 took the first 67 i/n out of it |
| 8 | **Unify the two threat systems.** We run *two* independent ones: the Wave-G board map for move ordering (`compute_threats`, ~2.1% of instructions, now lazy) and the NNUE FullThreats event stream (`record_square_touch` + `record_targets`, 3.55%). SF18 runs **one** (`update_piece_threats`, 5.65% of its instructions) that serves both its ordering and its net. Ours together are 397 i/n against SF18's 304 for the same two jobs. | ~90–160 i/n | yes | large; the two have different shapes (per-type attacked-square masks vs (attacker, from, to, victim) tuples) and the unification is a design exercise, not a patch |
| 6 | `l2_tail` + `l3` — 2.16% of the engine for a 16→32→1 tail (544 MACs) | ≤ 0.8% | f32 reorder | SF18's `fc_1`+`fc_2` are the same shape and, on the §1 tail assumption, ~1.4% of its engine — so this is only ~1.5×, the smallest ratio in the eval layer. Study 3's chain-split half is already rejected. **Lowest priority.** |
| 7 | `search::<NonPv>` outlining (carried item #1, "~5%") | see §7.3 | yes | **downgraded** — the cold mass is inert policy code that never executes |

Items 1+2 were written up in the first revision of this study as "a CENN format
rev, needs a retrain". **Both landed in this study without touching the trainer
or the file format** (§4.5, §4.6) for a combined −4.32%. The blocker was never
the net; it was insisting on a bit-exact gate for a quantisation change. What
they needed was a *fidelity* gate — `scripts/evaldiff.py`, 800 positions, cp
deltas — which is the instrument this study should have built on hour one.

---

## 6. THE FORMAT-REV ITEMS, MEASURED

### 6.1 Split quantisation is 24 of the 48 instructions in `activate`'s loop

SF18 carries the same two-accumulator shape we do — `HalfKAv2_hm` **and**
`FullThreats`, both 1024-wide, kept as separate `AccumulatorState`s and merged
at transform time (`nnue_feature_transformer.h:326-360`, sf_18 tag). The merge
is one instruction per lane group:

```
acc0a = vec_add_16(in0[…], tin0[…])      // psq + threat, same scale
sum0a = vec_slli_16(vec_max_16(vec_min_16(acc0a, One), Zero), 7)
…                                        // ~19 instructions per 64 output bytes
```

Ours cannot add directly, because the threat transformer is quantised at 64 and
the HalfKA one at 255. `activate_inner` must compute
`round(threat · 255/64)` with round-half-away-from-zero first — `vpaddsw`
(×2, to reach `4t`), `vpabsw`, `vpmulhrsw`, `vpcmplt` mask, `vpmask_sub` = **six
extra ops per merge**, four merges per loop body, **24 of the body's 48
instructions**. Per 64 output bytes: **ours 48, SF18 ~19 — 2.5×.**

Measurement (`uniquant` arm, deliberately eval-changing, reverted): replacing
the whole rescale with `signed = tv` — SF's exact shape — measured
**35,014,708,233 vs 35,991,058,878 instructions = −2.71%**. That arm's tree
differs (eval hash `56d6fe91…`, d13 total 1,961,283 vs 1,972,607), so treat it
as ±0.5%; the static accounting independently says 2.47%. **Quote ~2.5%.**

This is carried option #13 (D3 quant unification), and it now has a hard number
and a mechanism proven by side-by-side with the reference. It is **not**
bit-exact-able in the engine: pre-scaling the i8 threat rows to i16 gives
`Σ round(w·255/64)` where the runtime computes `round(Σ w · 255/64)`.

### 6.2 The skip lane is dense where SF's is one more sparse row

`skip_dot` reads all 1280 activated bytes (80 zmm iterations, `vpmovzxbd` +
`vcvtdq2ps` + `vfmadd132ps`) for **2.50% of all engine instructions**. SF18's
equivalent lane is `AffineTransformSparseInput<TransformedFeatureDimensions,
FC_0_OUTPUTS + 1>` — literally one extra output row of the same **sparse int8**
matmul it already runs, so it costs ~1/16 of `fc_0`. Quantising our `skipw` to
i8 and widening `L2` to 17 lanes would move 2.50% down to ~0.15%. Same format
rev as §6.1.

---

## 7. REJECTED / REFUTED, with the measurement that kills each

| claim / option | the measurement that kills it |
|---|---|
| **7.1 The insertion sort's bounds checks are worth ~0.7% of the engine** (`bcheck8.py` scored them 14.29% of `init_quiets`, top site 6.38%) | Restating the `sorted_end ≤ p` invariant as a prefix re-borrow + `min` removed one of sixteen panic edges and measured **−0.045%** (35,991,058,878 → 35,974,886,358, identical tree, static text unchanged at 1,426 lines). The 6.38% was **skid onto the compare after a store**. REVERTED. |
| **7.2 The threat-PSQ1 delta's out-param round-trip is 12.4% of `at_full`** (the two hottest stores in the whole eval layer) | Accumulating into a local `[i32; BUCKETS]` and writing back once measured **+1.306% instructions** (36,461,356,237 vs 35,991,058,878, identical tree, eval bit-exact). It is *worse*. The store share was store-to-load-forwarding skid, not a removable cost. REVERTED. |
| **7.3 `search::<NonPv>` outlining is worth ~5%** (impl-1 §5 #1, carried #1) | 24.9% of the symbol's 5,757 static instructions sit in cold runs ≥30 instructions, but `addr2line` on a line-table PGO build resolves the two largest (143 + 41 + 67 instructions) to the **inert policy consults** — `active_policy_head()` and `last_l2_for` under `#[cfg(feature = "sota")]` — which measure **0.000% dynamic** because no `.cpol` is installed and the gate is `tt_data.mv == NULL && depth >= 3`. Outlining them buys register pressure only, which is precisely impl-1's `update_root_move` case (**−0.010%**). NOT ATTEMPTED, and the "~5%" estimate is withdrawn pending a cold-mass analysis that finds *executing* register-hungry blocks. |
| **7.4 Sparse `skip_dot` over the nnz list** (carried option 6, "+1–1.5%") | FT group density is **31%**, not the 5–15% the source comment implies, so 16-element block occupancy is 77% and the ceiling is 0.6% — §3.2. The measurement is the `l1_dot` block's own instruction share (6.17% ÷ 11 instructions per group ÷ 0.366 evals/node = ~100 of 320 groups). DEAD. |
| **7.5 "`memset`/`memcpy` absent above 0.4%" (study 7 §6, B4 MOOT)** | `memmove-vec-unaligned-erms.S` is **~2.00% of all engine instructions** at D48X. It was invisible because perf cannot symbolise glibc ifunc bodies. The **memset** half of the original claim stands (0.10%), so the dead `[0u8; L1]` in `forward_into` is genuinely not worth removing — but B4-as-memmove is now a live 2% item with named call sites (§3.3). |
| **7.6 "The remaining deficit is the shell / `next_move` + `search` code size" (study 7 §0.6, option 3)** | Totalled rather than compared per symbol, our movepick is **667 i/n against SF18's 734** and our whole shell is **1.163×**. 59% of the excess is eval, 18% make/unmake, 14% shell. The shell is not where parity lives. |
| **1280/1024 = 1.25× as the justified width premium** | SF18's width-scaling instruction share is **32.03%**, so the justified premium is **1.080×**, not 1.25×. Assuming 1.25 would have hidden 660 i/n of real excess. |
| **`perf annotate` percent as a sizing instrument on this box** | Two arms selected by it (7.1, 7.2) delivered −0.045% and **+1.31%**. The one arm selected by static-instructions × trip-count delivered −1.75%. Sample share locates *phases*; it does not price *changes*. |
| **7.7 `skip_dot`'s 80-deep serial `vfmadd` chain is worth splitting** (carried option 6's other half, study 3 §6.11's "+1–1.5%") | A four-accumulator split was implemented — **battery-identical** (800-FEN integer-cp hash `53fc2de1…ad64` unchanged) and node-identical — and measured over **nine interleaved paired rounds**: 4 of 9 favoured it, median cycles/node ratio **1.023**, per-round range 0.818–1.111. Nothing. Same physics study 3 found when it split `l1_dot`'s chain: the kernel streams ~5 KB of f32 weights per eval and is **load-bound, not latency-bound**. REVERTED; the negative is now recorded in the kernel's own comment so it is not re-litigated. |
| **7.8 The threat-PSQ1 row load is split into xmm + 2× `vmovq` and wants a fixed-size array reference** | `psqt_add`/`psqt_sub` took `&table[f*8..(f+1)*8]`, a runtime-length slice; giving them a `&[i32; BUCKETS]` via `try_into` — the codebase's own idiom for exactly this — measured **+0.185% instructions** (37,068,196,310 → 37,136,819,178, identical 5,001,629-node tree, native non-PGO pair). REVERTED. Second time in this study that the 8-bucket PSQ1 accumulate looked like free money and cost instructions (see 7.2). |
| **7.9 The `l1_dot` raw-pointer trick generalises to the PSQ1 row** | It does not. Dropping the length check with a pointer cast measured **+2.2%** — the slice index is where LLVM gets the aliasing facts that keep the surrounding fold optimised. §4.4's safe `try_into` form, paired with a register-resident accumulator, is what pays. An instrument that won once is not a technique. |
| **7.10 The null-move / empty-delta parent→child copy in a fold span is waste** | It is a MEMOIZATION, not waste. Skipping it (tracking a per-perspective source ply and leaving the empty ply uncomputed) is bit-exact and node-identical and measured **+1.04%**: every later fold from that ply then walks back past it and re-folds a longer span, repeatedly, instead of paying one 2560 B copy once. Implemented on both the HalfKA and threat stacks; REVERTED. |
| **"the ~4.9% in the net format needs a retrain"** — this study's own §5, first revision | Wrong, and it was the most expensive wrong thing in the first write-up. Both items were implementable without touching the trainer: the frame ratio is a *constant* (§4.5, −2.43%) and the skip lane is *thirty lines in the loader* (§4.6, −1.89%). What they needed was not a retrain but a different gate — cp fidelity over 800 positions instead of a bit-exact hash. |
| **7.11 Store the threat rows pre-widened as i16, deleting the fold's `vpmovsxbw`** | The fold is `vpmovsxbw` + `vpaddw` per 32 lanes per row — literally half its inner work is widening i8 to i16 — so storing the table at i16 halves it. Implemented (loader widens once at parse; the threat fold then calls the existing `apply_i16_deltas_into`), bit-exact, tests green, and measured **−2.10% instructions**. It still LOSES, and only a cache measurement can see why — the table goes 78 MB → 155 MB against a 32 MB L3: <br><br>`instructions 7105.7 → 6956.5/node (ratio 0.979, 5/5 rounds favouring i16)`<br>`cache-misses    65.0 →   88.7/node (ratio 1.299, 1/5)`<br>`cycles        4989.9 → 5392.6/node (ratio 1.078, 1/5)`<br><br>REVERTED. This is the one place in the study where instructions and the truth disagree, and it is worth stating plainly: **an instruction count cannot arbitrate a change that trades memory for ALU.** `scripts/cache8.sh` exists for exactly this shape and should be reached for whenever a candidate makes a table bigger. It also answers a question the study kept asking sideways — *why* the trainer exports threats at i8 rather than in HalfKA's frame. Not accuracy. Footprint. |
| SMP scaling | still UNMEASURED; the box was quiet for 40 minutes of this session and contended for the rest. Unchanged since study 7. |

---

## 8. UNCERTAINTIES AND THREATS TO VALIDITY

1. **The box went from quiet to load ~10 mid-session** (a user-launched
   `archon` vs `stockfish18` match). Every cycles/NPS number after 15:55 is
   unusable and none is quoted as a claim. `instructions:u` on a fixed-node
   tree is deterministic and unaffected; every kept conclusion rides on it.
   The final three-arm panel is the cleanest demonstration of the difference
   this chain has produced: over six interleaved rounds the **per-round paired
   NPS ratio of `ours-base` vs SF18 ranged 0.585–0.777** — a 33% spread on a
   pair of byte-identical binaries searching byte-identical trees — while
   `instructions:u` on the same six rounds reproduced at **0.0000%** for all
   three arms. The contended-box cycles ratio even moved the *wrong* way
   across the kept change (1.493 → 1.515) purely from load drift. Anyone
   reading a speed number off this box without the paired-round protocol is
   reading noise.
2. `f_w` for SF18 rests on one judgement call: that ~5% of
   `Network<1024>::evaluate` is the width-independent `fc_1`/`fc_2` tail.
   Sweeping that assumption 0.90–1.00 moves the parity target only 4,657–4,693
   i/n (±0.4%), and §1.1's independent route agrees to 0.3%.
3. The `uniquant` arm changes the eval and therefore the tree; its −2.71% is
   contaminated by tree shape and is quoted with the static count (2.47%)
   beside it.
4. Instruction *attribution* by symbol is exact-per-sample but PGO inlines
   kernels into drivers, so a phase split inside a symbol
   (`annot8.py` boundaries) is my reading of the disassembly, not a compiler
   fact. The block totals are robust; the phase splits are ±1 boundary.
5. `find_nnz`'s 31% density is derived arithmetic (block share ÷ static loop
   length ÷ evals/node), not a direct count. A direct instrumented count would
   be worth 20 minutes to the next agent.
6. Nothing here was Elo-tested. At the house ~1 Elo per 1% NPS the kept change
   is worth roughly +1.8% NPS ⇒ ~2 Elo; that is below any match's resolution
   and is offered as an instruction fact, not an Elo claim.
7. SF18's small net was deliberately **not** credited as a width justification.
   If one argued it should be, `f_w` would fall and the parity target would get
   *stricter*, not looser.

---

## 9. VERDICT ON THE MANDATE

**Parity was NOT reached, and it is not reachable by patch work.**

* Target: **4,677 instructions/node** (SF18 × the measured 1.080× width premium).
* Before: **7,116.6**. After six kept changes: **6,546.0** (−8.02%), measured on
  the same three-arm matched panel as SF18's 4,329.6.
* Residual: **1.400×** the parity target — **23.4% of the gap removed**
  (570.6 of 2,439.6 i/n).
* Whole-engine ratio vs SF18: **1.644× → 1.512×** instructions,
  **2.020× → 1.936×** branches.

The residual is decomposed and named in §1.2 and §5. **None of it is
architectural in the sense of "1280 is wider than 1024"** — that is fully paid
for by the 1.080× premium, and both derivations of it agree. What remains is
implementation:

* **~4.9% of the engine (items 1+2 in §5) is a CENN format rev** — unify the
  threat quantisation and fold the skip lane into the sparse matmul. Both are
  training/export changes with an engine change that is mechanical, and both
  put us on exactly SF18's shape. This is the highest-value item on the board
  and it is not engine surgery.
* **~5.7% is `at_full`'s replay plumbing**, a stage SF18 does not have at all.
* **~6.1% is make/unmake at 2.26×** — never once studied in eight studies.
* **~2.0% is accumulator `memmove`** with the call sites now named.

Sum: those four are ~18.7% of the engine against the 31.5% parity still needs
after this study's 3.29%. Parity with SF18 at equal net width is therefore
*approachable* but would take the whole list plus the shell's 14%, and nothing
in this study suggests a single lever worth more than 5%.

**The honest shape of the answer**: the engine is not losing to Stockfish on
any one thing, and this study did not find a single lever worth more than 2.5%.
It found six worth 8.0% together, and the two largest were sitting behind a
gate rather than behind a difficulty — quantisation choices that a bit-exact
standard forbids and a *fidelity* standard permits, at a measured cost of
0.14 and 0.94 SF cp. What remains is ~5.7% in a threat-replay stage SF18 does
not have, ~6% in a make/unmake path nobody has profiled, ~2% in duplicated
threat bookkeeping, and ~3% in accumulator copies. Each is small; the parity
target only falls if all of them do.

### What I would do next, in order

The four items below are what is left of the +2,438 i/n, after this study took
570 of it. Nothing here needs a retrain either — that assumption cost this
study its first revision (§7, last row).

1. **`at_full`'s replay plumbing** — ~5.7% of the engine, and SF18 has no
   equivalent stage at all. Its fold half is now known to be at its floor: it
   is `vpmovsxbw` + `vpaddw` per 32 lanes per row, and the obvious way to
   delete the widen — storing the rows at i16 — was implemented and **loses on
   cache** (§7.11). So the target is the plumbing half: `build_deltas`
   regenerating events from a snapshot, then `resolve_into` computing a
   perspective-relative index per event.
2. **Unify the two threat systems** (§5 item 8). We maintain the Wave-G board
   map *and* the NNUE FullThreats event stream separately, 397 i/n for the two,
   against SF18's single `update_piece_threats` at 304 for the same two jobs.
   §4.3 showed the ordering half is wanted by ~52% of nodes and the NNUE half
   by 37%; one lazily-resolved threat delta serving both is the shape SF has
   and we do not.
3. **Study make/unmake.** Still ~2.2× SF18 after §4.3 took 67 i/n out of it, on
   a subsystem no study in this chain has profiled. `Worker::do_move` is
   10,190 B of text and 2.38× SF's `do_move`.
4. **The accumulator `memmove` block**, now 3.18% of a smaller engine. The
   Finny cached refresh computes its accumulator, stores it to the cache, then
   `memcpy`s it to the stack slot; a two-destination kernel stores it once to
   each. Note §7.10 first: the *other* copies in that block are memoizations
   and removing them costs more than it saves.
5. **Re-run the matched counter panel on an idle box** and finally take the SMP
   scaling measurement, which is now four studies overdue.

## 10. ARTIFACTS (`perf-artifacts-8/`)

* `counters-baseline-wl50.txt` — the six interleaved matched fixed-node rounds.
* `arms.txt` — every arm's eval hash, d13 node total, and instruction count.
* `block-decomposition.txt` — the full block/branch/static-size decomposition.
* `flat-{ours-d48x,sf18}-{instructions,branches}.txt` — the four flat profiles.
* `annot-*.txt` — per-symbol disassembly with instruction-event weights.
* `scripts/`
  * `pstat8.sh` / `parse8.py` — interleaved matched fixed-node counter rounds.
  * `flat8.sh` — instruction- (or branch-) attributed flat profiles.
  * `annot8.py` — split a symbol into phases by address; opcode mix.
  * `cold8.py` — cold static mass inside a hot symbol (outlining candidates).
  * `spill8.py` / `spillall.sh` — the stack-traffic (register-pressure) signature.
    **Note**: `%rbp` is an ordinary GPR in this profile, so a spill detector must
    match `(%rsp` only — accepting `%rbp` reported `make_move_with` at 64%.
  * `bcheck8.py` — bounds-check inventory. **Its percentages are skid; use it as
    a candidate list only (§4.2).**
  * `libcalls8.py` — attribute libc `mem*` call sites to the enclosing symbol.
  * `where8.sh` — `addr2line` a list of text addresses (needs a line-table build:
    `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only` with the same
    `-C profile-use=` profdata; plain `-C debuginfo=1` in `RUSTFLAGS` is
    overridden by the profile and yields no line table).
  * `arm8.sh` — build one PGO arm, run both identity pins, measure instructions.

**Reproduction.**

```bash
# identity pins for the study binary + canonical net
./engines/bin/archon-s8-base nnue-eval engines/canonical/net.cenn \
  < tools/nnue/calib/wl50.fens | sha256sum        # -> d82caf8a…8163
taskset -c 4 ./engines/bin/archon-s8-base bench 16 1 13 \
  tools/nnue/calib/wl50.fens depth engines/canonical/net.cenn   # -> 1,972,607

# the matched counter panel
bash docs/research/perf-artifacts-8/scripts/pstat8.sh 6 300000 \
  tools/nnue/calib/wl50.fens /tmp/counters.txt \
  "ours:./engines/bin/archon-s8-base @BENCH@ engines/canonical/net.cenn" \
  "sf18:./engines/bin/stockfish18 @BENCH@"
python3 docs/research/perf-artifacts-8/scripts/parse8.py /tmp/counters.txt sf18

# one optimization arm, end to end
bash docs/research/perf-artifacts-8/scripts/arm8.sh <name>
```
