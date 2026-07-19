# ARCHON profiling study, round 2 — after opt #1 (vector head) + opt #2 (fused folds)

Date: 2026-07-19. Round 1 is `docs/research/profiling-study-2026-07-18.md`; this
round re-profiles the CURRENT sota binary (opt1 `8287e90` + opt2 `1cc2147`,
HEAD `b55260b` — docs-only commits since) and re-ranks the optimization list.
Study only; no engine code was modified.

**Headline:** both landed opts hit exactly where round 1 predicted, and harder
than the under-load measurements suggested — the clean-box package is
**+20.7% NPS at bit-identical nodes** (712k → 859k, pinned median-of-3,
depth-12 bench). The scalar f32 head is *completely gone* (0 scalar FMAs in
the whole binary, was 448; I1 miss rate 1.12% → **0.29%**), the fold memmove
is gone (9.4% → 2.9% residual with different, smaller causes). IPC rose to
**2.55**. The engine is no longer front-end/icache-bound; it is now bound by
**data-side latency in the threat-fold weight-row streaming** (`i8_deltas_into`,
16.4% of cycles, 27.4% of all D1 read misses) and by **branch pathology in the
threat observer** (19.9% block, 44% of all conditional mispredicts, 40% of
indirect). At the common position ARCHON sota now runs **0.90× Reckless**
(same net shape; was 0.80×) and 1.09× the SSE2 SF16 build. A second low-risk
package (prefetch, observer constants, copy elimination, static dispatch +
wide head kernel) worth an estimated **+8–14% NPS** is identified.

---

## 0. Methodology deltas vs round 1

- Same box (Ryzen 7 7700X, Zen 4, WSL2), same perf binary
  (`/usr/lib/linux-tools-6.8.0-136/perf`), same valgrind AVX2-lane workaround
  (EVEX undecodable; kernel dispatch falls back to AVX2 under valgrind's
  virtual CPUID — call counts are ISA-independent).
- **The box was quiet this time** (load < 1 throughout; no relabel fleet, no
  fastchess — checked before every heavy step). Round-1 absolutes were
  contention-deflated ~2×; therefore **all cross-engine rows were re-measured
  fresh** and round-1 absolute NPS/cycle numbers are NOT compared directly.
  Residual ~±2% run-to-run drift observed (concurrent doc-reading agent);
  mitigated with pinned (`taskset -c 14`, SMT sibling idle) median-of-3 runs.
- Builds (scratchpad target dir, HEAD source):
  - release, `-C target-cpu=x86-64-v3`, `--features sota` — **byte-identical
    to the shipped `data/nnue/experiments/opt2-equaltime/sota-opt2`** (cmp).
  - profiling profile + frame pointers (perf-record lane; 828 vs 860 knps —
    shares quoted from it, absolutes from release). Neither is PGO'd.
  - a `-C target-cpu=native` release, built only for the head-width check
    (§1, finding F4).
- Workloads identical to round 1: depth-12 bench for perf (3,515,701 nodes —
  bit-identical to round 1, confirming the opts preserved search), depth-8
  (147,691 nodes) for callgrind/cachegrind with the same cache geometry.
- Reckless still cannot be pinned (its NUMA layer asserts under taskset); its
  bench counters still include helper-thread spin. The common-position UCI
  protocol (§4) is the fair cross-engine comparator.

---

## 1. Did the fixes land where round 1 predicted? (verification table)

| round-1 finding | claim | round-2 evidence | verdict |
|---|---|---|---|
| §4.2 scalar f32 head tail: 448 scalar `vfmadd*ss`, 12.6% self, 5,726-instr function, 28% of all I1 misses | opt #1 vectorizes it, bit-identical | `objdump` of the release binary: **0 `vfmadd*ss` in the entire binary**; `forward`+`propagate_*` now fully inlined into `SotaContext::evaluate` — **820 instructions total** (vs 5,726 for `forward` alone); 34 packed FMAs (24 `vfmadd231ps` + 10 `vfmadd213ps`). Cachegrind I1 misses **44.25M → 11.00M (−75%), rate 1.12% → 0.29%** — even better than the predicted 28% cut, because the giant function also evicted everyone else. `forward` no longer appears in the perf profile at any threshold. | **FIXED, over-delivered on icache** |
| §4.3 fold memmove: `__memmove_avx512` 9.4% of cycles, 1.5 KB parent→child copies at both fold sites | opt #2 fuses copy+apply | perf: memmove **9.4% → 2.86%**, and its callers are no longer the fold sites (callgrind: zero memcpy from `HalfkaContext::compute` / `ThreatContext::compute` fold loops; both now call `*_into` kernels — `sota_accumulator.rs:100-108`, `sota_threats.rs:722-729`). D-refs down 1.35G → 1.19G (−12%) at depth 8. Residual 2.86% has different causes (§2.1). | **FIXED** |
| package estimate +17–24% (items 1+2) | — | clean-box A/B at identical nodes: 712,112 → 859,164 nps = **+20.7%** (commit-time under-load: +17.1%); NPS-harness geomean now 1,076,765 | **CONFIRMED** |
| §3 "latency/front-end bound, IPC 1.39" | — | IPC now **2.55** (57.53G instr / 22.60G cycles, d12), branch-miss 1.94% (was 2.17%) | front-end problem solved |
| §4.4 observer 15.0%, 214k indirect mispredicts | untouched by opts | 19.9% of the faster binary; cachegrind indirect mispredicts in `push_sota_dirty` = **214,511 — the identical count** (code unchanged, tree bit-identical) | unchanged, now relatively bigger |

Two sub-findings on the landed opts:

- **F4 — the vectorized head is stuck at 128-bit.** The autovectorized
  `propagate_l2/l3` compile to xmm-width FMA (96 xmm operands vs 2–4 ymm/zmm)
  with heavy `vmovaps` register shuffling (perf annotate). A
  `-C target-cpu=native` rebuild does NOT widen it (same 96 xmm; bench
  869/877/840 knps ≈ +1% ≈ noise) — LLVM's SLP cost model, not the target,
  is the limiter. Full width needs an explicit kernel (§5 item 4).
- **F5 — the `KERNELS` LazyLock check survives per call.** Release disassembly
  of `evaluate` shows `mov KERNELS+0x30,%eax; test; jne` before each of the 4
  fn-pointer kernel calls per eval (`sota_simd.rs:31-48` dispatch): round-1
  item #4 is unchanged.

---

## 2. New hot-path breakdown (perf, depth-12, 4,474 samples, self-time)

One-time startup ≈ 3.7–4.1% of this profile (net parse `load_v2_bytes` 1.39 +
`read_i8s` 0.68 + `read_i16s` 0.28, `init_magics` 0.40, startup memset/page-in
~1.0) — excluded from the subsystem shares below (divide by 0.96).

| share | subsystem | functions (`file:line`) |
|------:|-----------|--------------------------|
| **17.1%** | **threat fold** (new #1 function) | `avx512::i8_deltas_into` 16.42% (`sota_simd.rs:563-588`, driven by `sota_threats.rs:700-732`) + wrapper |
| **20.7%** | **threat observer** (new #1 block) | `push_sota_dirty` self 15.12% (`eval/mod.rs:206-216` with `resolve_entry`/`build_deltas`/`threat_index` inlined; `sota_threats.rs:762-794, 800-897`, `sota_features.rs:324-348`) + `record_square_touch` 3.08% (`sota_threats.rs:196-327`) + `record_outgoing` 1.01% + `ripple` 0.67% |
| 9.2% | HalfKA fold | `i16_deltas_into` 7.36% (`sota_simd.rs:538-561`, driven by `sota_accumulator.rs:96-110`) + in-place `i16_deltas` 1.50% (Finny diffs) |
| 17.1% | head forward | `activate` 5.84% (`sota_simd.rs:615-669`) + `SotaContext::evaluate` self 5.18% (the *inlined* vectorized head + orchestration, `sota_head.rs:129-225`) + `l1_dot` 4.01% (`sota_simd.rs:590-613`) + `find_nnz` 1.41% |
| 2.4% | Finny refresh | `finny_refresh` 2.28% (`sota_accumulator.rs:113-168`) |
| 2.0% | threat fold glue | `ThreatContext::compute` 1.88% |
| 2.5% | residual memmove/memset | §2.1 — no longer the fold copies |
| **71.0%** | **NNUE total** (was ~70% in round 1 — see note) | |
| 10.0% | move ordering (picker) | `next_move` 6.82% + `new` 2.84% (`movepick.rs:346, 257`) |
| 7.0% | make/unmake | `make_move_with` 4.39% + `set_check_info` 1.34% + `gives_check`/`do_move`/`attackers_to` |
| 3.2% | search control | `search` 2.24% + `qsearch` 0.83% |
| 2.9% | movegen/attacks/SEE | `attacks` 0.76% + `see_ge` 0.83% + `legal` 0.40% + misc |
| 1.4% | history | `correction_value` 1.01% + updates |
| 0.4% | TT | `probe` 0.38% |
| **29.0%** | **search machinery total** | |

Note on the ~unchanged 70/30 split: round-1's share table came from a
*contended* box, which inflated the memory-bound NNUE parts; the opts then
removed ~17–20% of cycles almost entirely from NNUE. The two effects roughly
cancel in the *shares*; the absolute win is proven by the node-identical
+20.7%. Do not read the stable split as "the opts didn't land."

The laziness anatomy is byte-for-byte intact (callgrind, d8): 50,192 evals =
0.34/node; `activate`/`find_nnz`/`l1_dot` exactly once per eval; fused HalfKA
folds `i16_deltas_into` = 107,704 = **exactly** round-1's 125,869 i16 folds
minus the 18,165 Finny-path applications that stayed in-place. Threat folds
unchanged at ~2.67/eval.

**The single hottest instruction in the engine** is now in the observer:
`threat_index`'s `lut1` load + sentinel compare (`sota_features.rs:342-344`,
`cmp $0xed2f,%edx`) takes **21.15% of `push_sota_dirty`'s samples ≈ 3.2% of
total runtime** — a dependent 4-byte table load consumed by a data-dependent
exclusion branch, executed per raw event (~15/move) per perspective. Note
also that `threat_index` re-derefs the `THREAT_TABLES` LazyLock **per event**
(`sota_features.rs:332`).

### 2.1 Where the residual 2.86% memmove lives (callgrind caller attribution)

301,323 libc memcpy calls at depth 8 — **6.0 per eval**, none from the fold
paths anymore:

| calls (d8) | caller | what it copies |
|---:|---|---|
| 118,455 (2.36/eval) | `SotaContext::evaluate` | by-value buffer returns: `activate_ft` `[u8;768]` (768 B, `sota_head.rs:82-86`) + `find_nnz` `([u16;192],usize)` (384 B, `:119-123`) every eval, + the 1,536 B `finny_refresh` return at 0.36/eval |
| 72,660 (4.0/refresh) | `finny_refresh` | `let mut acc = cached.acc` + return-by-value + `FinnyEntry` write-back + stack-slot store — **4 × 1.5 KB per refresh** (`sota_accumulator.rs:127, 147, 150-151, 161-167`) |
| 53,283 | `MovePicker::new` | movelist init (`movepick.rs:257`) |
| 23,128 | `i8_deltas_into` | **null-move folds**: empty add/sub list ⇒ the fused kernel degenerates to a pure 1.5 KB copy (`search/mod.rs:2470-2474` pushes `DirtyPiece::EMPTY`) |
| 14,182 + tail | `iterative_deepening`, sort, etc. | root-move bookkeeping |

---

## 3. Cache & branch analysis (cachegrind, depth-8, AVX2 lane)

| metric | round 1 | round 2 | reading |
|---|---|---|---|
| I1 miss rate | 1.12% (44.25M) | **0.29% (11.00M)** | **fixed** — opt #1's main structural win |
| D1 miss rate | 5.2% (70.2M) | 5.7% (67.2M) | unchanged absolute; now the dominant class |
| LL(32MB) miss rate | 0.1% (5.05M) | 0.1% (5.05M) | still not DRAM-bound; hot rows live in L2/L3 |
| cond mispredict | 2.5% | 2.4% (12.7M) | fine globally, concentrated (below) |
| indirect mispredict | 16.2% | 18.3% (533k) | same two culprits |

Where the misses are now:

- **D1 read misses are the threat/HalfKA row streams**: `i8_deltas_into`
  13.96M = **27.4%** of all D1 rd misses (+ 2.62M = 16.2% of D1 wr, + 8.0% of
  LL rd), `i16_deltas_into` 8.15M = 16.0%, `l1_dot` 5.99M = 11.8%. Identical
  physics to round 1 (~15 × 768 B i8 rows scattered over 46.6 MB per fold has
  no temporal locality); the difference is that with the head and memmove
  fixed, **this is now the #1 cycle sink** — perf annotate shows
  `i8_deltas_into`'s samples piled on the `vpaddw/vpsubw` that consume each
  `vpmovsxbw` row load (18.1% + 9.7% + 7.7% on three such pairs): classic
  exposed load-use latency, the textbook software-prefetch target.
- **Branch pathology is the observer**: `push_sota_dirty` owns 3.66M = 28.8%
  of ALL conditional mispredicts + 214,511 = 40.2% of ALL indirect mispredicts
  (the per-event piece-type dispatch + the O(n²) cancel scan + the
  `threat_index` exclusion branch); `record_square_touch` adds another 15.4%
  of cond. Observer total: **~44% of the program's conditional mispredicts.**
- **MovePicker is the new top icache consumer** (1.30M = 11.8% of the now-tiny
  I1 misses, plus 22.0% of indirect mispredicts) — nothing pathological, but
  it is where the *remaining* front-end cost sits (`movepick.rs:346`).
- Zen-4 PMU cross-check (native lane): L1d-load-miss 14.31% of loads (the AMD
  event counts speculative/prefetch traffic — higher than the model, same
  direction), L1i misses collapsed to 20.9M over the whole 4.3 s run.

**Verdict:** compute is now healthy (IPC 2.55); the binary is
**data-latency-bound in the folds and branch-bound in the observer**. Round
1's "front-end/icache" diagnosis no longer applies — code-layout tools (PGO)
will pay less than they did pre-opt; prefetch + branch surgery pay more.

---

## 4. Cross-engine NPS / efficiency (all re-measured on the quiet box)

Own-bench suites (perf stat, pinned core 14, warm — SF18's cold first run
discarded):

| engine (build ISA) | eval arch | knps | instr/node | cycles/node | IPC | br-miss |
|---|---|---:|---:|---:|---:|---:|
| ARCHON sota opt1+2 (v3 + AVX-512 kernels) | 768 HalfKA+threats, hybrid head | **860** (d12) | 16.0k¹ | 6.1k¹ | 2.55 | 1.94% |
| ARCHON classic-2048 (native+PGO) | 2048 classic, 1-layer | **2,106** (d12) | 5.2k | 2.7k | 1.92 | 2.61% |
| SF16 (distro **SSE2**) | 2048 HalfKAv2_hm | **1,924** (d13) | 8.9k | 3.0k | 2.98 | 3.57% |
| SF18 (local **AVX2**) | 1024 HalfKAv2_hm + FullThreats | **1,491** | 7.5k | 4.7k | 1.60 | 7.66% |
| Reckless d445b8b (native) | 768 PST+threats, same head shape | **1,391**² | ≤11.7k² | ≤8.0k² | 1.46² | 5.37% |
| Integral v6 | 1536 HalfKA, no threats | **2,121** | 4.2k | 3.4k | 1.23 | 3.77% |

¹ net of ~1.66G startup instructions (callgrind-measured loader+magics share).
² unpinned + helper-thread spin in the counters; treat as upper bounds.

Common position — the fair same-work comparator (startpos, `go nodes 500000`,
cumulative nps of the final info line, 2 samples):

| engine | knps | vs ARCHON sota |
|---|---:|---:|
| ARCHON sota (opt1+2) | **952–959** | 1.00 |
| Reckless | 1,047–1,064 | **0.90×** (round-1: 0.80×) |
| SF16 (SSE2) | 870–884 | 1.09× in ARCHON's favor |
| SF18 (AVX2) | 1,150–1,161 | 0.83× |
| ARCHON classic-2048 | 1,890–1,945 | 0.49× |
| Integral v6 | 2,197–2,242 | 0.43× |

Readings:

- **The same-shape gap to Reckless halved**: 0.80× → 0.90×. The remaining
  ~10% is well accounted for by §5 items 1–4 (prefetch, observer, copies,
  dispatch+head-width) — i.e. the round-1 thesis ("nothing architectural,
  pure constant-factor headroom") continues to verify as the items land.
- **ARCHON sota now outruns the SSE2 SF16 build at the same position** while
  computing a bigger feature set — but SF16's *distro build* is the weakest
  fair point; the instruction-efficiency king is still SF18: **7.5k
  instr/node vs our 16.0k (2.1×, was 2.3×)**. The ratio barely moved because
  opt #2 removed *cycles*, not instructions (ERMS `rep movsb` was ~free in Ir),
  and opt #1's Ir savings were ~1k/node. The remaining 2× is fold volume
  (~15 threat features/fold by construction), the always-paid observer, the
  i32 activate tax, and the deeper head — format/architecture items, not
  kernel bugs.
- Integral remains the raw-speed outlier (4.2k instr/node, no threat features,
  PEXT + per-tier PGO builds) — same shape as round 1, now measured clean.

---

## 5. Ranked optimization list (post-opt1+opt2)

"NPS" = pure speed, bench-node-identical. Elo per the D39 house rule (~+9–10
internal Elo per +10% speed at 10+0.1; opt #1's measured +28.8±10.3 at
+13–15% NPS actually beat the rule). Round-1 numbering in brackets.

| # | what / where | expected gain | effort | risk |
|---|---|---|---|---|
| **1** [was #3] | **Prefetch the fold weight rows.** At `i8_deltas_into`/`i16_deltas_into` entry, `_mm_prefetch` every add/sub row (i8 rows are 768 B = 12 lines each, ~15 scattered over 46.6 MB) before the lane loop; optionally process two features per pass for MLP (Reckless's `double_dpbusd` shape). `sota_simd.rs:538-588` (+ AVX2 twins `:404-475`). Evidence: 16.4% of cycles with samples on the load consumers, 27.4% of D1 rd misses, 8% of LL rd. **Round-1 deprioritized this; that call is now wrong** — it is the single highest-value pure-speed item left. | **+3–6% NPS** | LOW | LOW (no math change; node-identical) |
| **2** [was #5] | **Observer constant factor.** (a) hoist the `THREAT_TABLES` LazyLock deref out of the per-event path (`sota_features.rs:332` — once per `resolve_entry`, not per event); (b) precompute per-(attacker, perspective) `lut1` row pointers to shorten the dependent-load chain feeding the 3.2%-of-runtime sentinel compare (`sota_features.rs:342-346`); (c) branchless piece-type dispatch in `record_square_touch` (`sota_threats.rs:200-242`; observer = 44% of all cond + 40% of all indirect mispredicts); (d) replace `resolve_entry`'s O(n²) cancel scan (`sota_threats.rs:776-789`). The 19.9% block; 20–30% of it shavable. | **+3–5% NPS** | MED | LOW-MED (`threats_replay_*` parity gates exist) |
| **3** [NEW — successor of opt #2] | **Kill the residual eval-path copies (2.9%, 6 memcpys/eval).** Pass caller buffers into `activate_ft`/`find_nnz` instead of by-value returns (`sota_head.rs:82-86, 119-123, 211-225`); make `finny_refresh` write the stack slot + cache entry in place instead of 4×1.5 KB round trips (`sota_accumulator.rs:127-167`); short-circuit the empty-delta (null-move) fold to a straight copy or an alias tag (23,128 pure-copy kernel calls at d8). All bit-identical plumbing. | **+1.5–2.5% NPS** | LOW | LOW |
| **4** [was #4, merged with F4] | **Static kernel dispatch + explicit wide head kernel.** The per-call LazyLock check is confirmed alive (`mov KERNELS+0x30; test; jne` ×4/eval, `sota_simd.rs:31-48`); replace with one-time-detected static dispatch (or `cfg` modules per D16 fast-build). Fold the head in: the autovectorized tail is stuck at xmm width even at `target-cpu=native` (F4), so give `propagate_l1/l2/l3` an explicit ymm/zmm kernel behind the same dispatch — preserving per-`out[j]` accumulation order keeps it bit-identical (same argument as opt #1). | **+2–4% NPS** | MED | LOW |
| **5** [was #7] | **Bulk net parse.** Unchanged per-element readers (`sota_loader.rs:512-543`) cost 1.08G Ir and a measured **0.15 s per process spawn** (bench-d1 wall 0.18s vs 0.11s no-net baseline); at depth-8 profiling scale that's 44% of all instructions in the run. No Elo at long TC; real QoL for SPRT spawns and every future short-workload profile. | startup −0.13 s | TRIVIAL | none |
| 6 [was #8] | **Unify threat/FT quantization in the next CENN rev** → `activate` collapses from the i32 widen+rescale merge (5.8% now, relatively bigger than round 1) to the standard i16 `mulhi` pairwise form. Needs trainer/netpack rev; not bit-exact vs current nets. Bundle with the next net-format change. | +2–3% NPS | MED-HIGH | MED |
| 7 [was #9] | **MovePicker constant factor** — 10.0% of cycles, now the top I1 consumer (11.8% of misses) and 22% of indirect mispredicts (`movepick.rs:257, 346`). Coordinate with the search study (it is entangled with staged-generation policy). | +1–3% NPS | MED | MED |
| 8 [was #10] | **Huge pages for TT + net** — unchanged rationale; measured dTLB misses are modest (9.7M/4.8 s). Relevant at real hash sizes. | +0–2% | LOW | LOW |
| 9 [was #6] | AVX-512 backend for the classic v1 kernels — only if the classic lane keeps shipping. | +8–12% (classic only) | LOW-MED | LOW |
| — | **Demoted: PGO/code-layout work as a speed lever.** Round-1 justified PGO's +10% largely via the I1 catastrophe; with I1 at 0.29% the layout upside is much smaller. Keep PGO for releases, don't count on it for wins. | — | — | — |

**What the landed opts changed about priorities:** the profile flipped from
front-end-bound to memory-latency/branch-bound, which (a) promotes prefetch
(item 1) from "deprioritized" to #1, (b) promotes the observer (item 2) —
its absolute cost is untouched and its share grew to ~20%, (c) spawns item 3
(the copies the fused kernels *didn't* cover — by-value returns and the Finny
path), and (d) demotes anything icache-flavored.

**Strategic read:** items 1–4 are independent, bench-node-preserving, and sum
to **+8–14% NPS** — roughly the remaining same-shape gap to Reckless (0.90× →
~parity). Beyond that, the leftover ~2× instr/node vs SF18 is architecture and
format (fold event volume, always-paid observer schedule — forced by sibling
board destruction, module docs `sota_threats.rs:36-46` — the i32 activate
merge, head depth): those are net-format-rev or design decisions, not kernel
fixes, and the marginal Elo per engineering-day of pure-speed work drops
sharply after this second package. Per D39, +8–14% NPS ≈ +7–13 internal Elo
at 10+0.1 (external ~half), against the −42.8±? SF16 equal-time gap.

---

## Appendix: raw data

Scratchpad `prof2/` (`/tmp/claude-0/-root-projects-learnrust-chessengine/9162696c-.../scratchpad/prof2/`):
`sota-r2-rel` (release v3, ≡ shipped sota-opt2), `sota-r2-prof` (profiling),
`sota-r2-native`, `sota-r2-d12.perf.data` (4,474 samples), `cg-r2-d8.out`
(callgrind), `ccg-r2-d8.out` (cachegrind), `sota-r2-rel.asm` / `evaluate.asm` /
`evaluate-native.asm` (disassembly). Baseline binary:
`data/nnue/experiments/opt2-equaltime/sota-base`. perf:
`/usr/lib/linux-tools-6.8.0-136/perf` (direct invocation).

Key raw numbers: package A/B d12 pinned (base 726/712/708 knps, opt
860/859/853 knps, nodes 3,515,701 both); perf stat opt d12: 22.604G cycles,
57.532G instr, 7.712G branches, 149.6M br-miss; cache pass: 22.348G L1d loads,
3.197G L1d misses, 20.9M L1i misses, 9.7M dTLB, 0.8M iTLB; cachegrind d8:
Ir 3.733G, I1 11.00M/0.29%, D1 67.2M/5.7%, LL 5.05M/0.1%, Bc 533.8M/12.7M
(2.4%), Bi 2.92M/533k (18.3%); NPS-harness geomean 1,076,765
(median 1,054,600).
